#include <iostream>
#include <string>
#include <cstring>
#include <csignal>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include "rkllm.h"
#include "httplib.h"

using namespace std;

LLMHandle llmHandle = nullptr;
static mutex g_infer_mutex;
static httplib::DataSink* g_active_sink = nullptr;
static atomic<bool> g_is_finished(false);

// Telemetrické proměnné
static chrono::steady_clock::time_point g_start_time;
static chrono::steady_clock::time_point g_first_token_time;
static int g_token_count = 0;
static bool g_got_first_token = false;

string escape_json(const string& s) {
    ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else o << c;
    }
    return o.str();
}

void send_sse_chunk(const string& text) {
    if (!g_active_sink || text.empty()) return;
    string escaped = escape_json(text);
    string sse = "data: {\"id\":\"chatcmpl-npu\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + escaped + "\"}}]}\n\n";
    g_active_sink->write(sse.data(), sse.size());
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        if (!g_got_first_token) {
            g_first_token_time = chrono::steady_clock::now();
            g_got_first_token = true;
        }
        if (result && result->text) {
            string chunk = result->text;
            if (!chunk.empty()) {
                g_token_count++;
                send_sse_chunk(chunk);
            }
        }
    } else if (state == RKLLM_RUN_FINISH) {
        auto end_time = chrono::steady_clock::now();
        double total_time_s = chrono::duration<double>(end_time - g_start_time).count();
        double ttft_ms = g_got_first_token ? chrono::duration<double, milli>(g_first_token_time - g_start_time).count() : 0.0;
        double speed_tok_s = (total_time_s > 0) ? (g_token_count / total_time_s) : 0.0;

        ostringstream telemetry;
        telemetry << fixed << setprecision(2);
        telemetry << "\n\n---\n"
                  << "*⚡ [RK3588 NPU Telemetrie] "
                  << "Vygenerováno: " << g_token_count << " tok | "
                  << "TTFT: " << ttft_ms << " ms | "
                  << "Čas: " << total_time_s << " s | "
                  << "Rychlost: **" << speed_tok_s << " tok/s***";

        send_sse_chunk(telemetry.str());

        if (g_active_sink) {
            string done = "data: {\"id\":\"chatcmpl-npu\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            g_active_sink->write(done.data(), done.size());
        }
        g_is_finished.store(true);
    } else if (state == RKLLM_RUN_ERROR) {
        cerr << "[NPU Error] RKLLM_RUN_ERROR!" << endl;
        g_is_finished.store(true);
    }
    return 0;
}

void cleanup_and_exit(int sig) {
    if (llmHandle) {
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
    exit(0);
}

// Sestavení ChatML šablony z JSON těla požadavku
string build_chatml_prompt(const string& json_str) {
    string system_msg = "";
    string user_msg = "";
    
    // Extrakce system zprávy
    size_t sys_pos = json_str.find("\"system\"");
    if (sys_pos != string::npos) {
        size_t c_pos = json_str.find("\"content\"", sys_pos);
        if (c_pos != string::npos) {
            size_t start = json_str.find("\"", json_str.find(":", c_pos)) + 1;
            size_t end = json_str.find("\"", start);
            if (start != string::npos && end != string::npos) {
                system_msg = json_str.substr(start, end - start);
            }
        }
    }

    // Extrakce poslední user zprávy
    size_t user_pos = json_str.rfind("\"user\"");
    if (user_pos != string::npos) {
        size_t c_pos = json_str.find("\"content\"", user_pos);
        if (c_pos != string::npos) {
            size_t start = json_str.find("\"", json_str.find(":", c_pos)) + 1;
            size_t end = json_str.find("\"", start);
            if (start != string::npos && end != string::npos) {
                user_msg = json_str.substr(start, end - start);
            }
        }
    }

    if (user_msg.empty()) {
        size_t prompt_pos = json_str.find("\"prompt\"");
        if (prompt_pos != string::npos) {
            size_t start = json_str.find("\"", json_str.find(":", prompt_pos)) + 1;
            size_t end = json_str.find("\"", start);
            if (start != string::npos && end != string::npos) {
                user_msg = json_str.substr(start, end - start);
            }
        }
    }

    if (user_msg.empty()) user_msg = "Ahoj";

    // Korektní Qwen ChatML šablona
    string prompt = "";
    if (!system_msg.empty()) {
        prompt += "<|im_start|>system\n" + system_msg + "<|im_end|>\n";
    } else {
        prompt += "<|im_start|>system\nJsi expertní programátor a asistent.<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Pouziti: ./server <model_path>" << endl;
        return 1;
    }

    signal(SIGINT, cleanup_and_exit);
    signal(SIGTERM, cleanup_and_exit);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.max_context_len = 4096;
    param.max_new_tokens = 2048;
    param.top_k = 1;
    param.top_p = 0.9f;
    param.temperature = 0.3f;
    param.repeat_penalty = 1.15f;
    param.skip_special_token = true;

    cout << "[NPU Server] Inicializace RKLLM..." << endl;
    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret != 0) {
        cerr << "[NPU Server] Chyba init: " << ret << endl;
        return 1;
    }
    cout << "[NPU Server] Model pripraven v NPU." << endl;

    httplib::Server svr;

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"ai-mentor\",\"object\":\"model\"}]}", "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        string formatted_prompt = build_chatml_prompt(req.body);
        cout << "\n[NPU Server] Vyhodnocuji prompt:\n" << formatted_prompt << endl;

        res.set_chunked_content_provider(
            "text/event-stream",
            [formatted_prompt](size_t offset, httplib::DataSink &sink) mutable {
                lock_guard<mutex> lock(g_infer_mutex);
                g_active_sink = &sink;
                g_is_finished.store(false);
                g_token_count = 0;
                g_got_first_token = false;
                g_start_time = chrono::steady_clock::now();

                RKLLMInput input;
                memset(&input, 0, sizeof(RKLLMInput));
                input.input_type = RKLLM_INPUT_PROMPT;
                input.prompt_input = const_cast<char*>(formatted_prompt.c_str());

                RKLLMInferParam infer_param;
                memset(&infer_param, 0, sizeof(RKLLMInferParam));
                infer_param.mode = RKLLM_INFER_GENERATE;

                int run_ret = rkllm_run(llmHandle, &input, &infer_param, nullptr);
                if (run_ret == 0) {
                    while (!g_is_finished.load()) {
                        this_thread::sleep_for(chrono::milliseconds(2));
                    }
                } else {
                    cerr << "[NPU Server] rkllm_run chyba: " << run_ret << endl;
                }

                g_active_sink = nullptr;
                sink.done();
                return true;
            }
        );
    });

    cout << "[NPU Server] Posloucham na portu 8088..." << endl;
    svr.listen("0.0.0.0", 8088);

    cleanup_and_exit(0);
    return 0;
}
