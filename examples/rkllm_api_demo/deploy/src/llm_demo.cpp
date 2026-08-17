#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem> // C++17 pro práci se složkami

using namespace std;
using Clock = std::chrono::high_resolution_clock;

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"
#define RED     "\033[31m"
#define CYAN    "\033[36m"

LLMHandle llmHandle = nullptr;
static Clock::time_point infer_start;
static int chunk_count = 0; // Přejmenováno z token_count
static int total_queries = 0;
static bool is_pipe = false;
static string current_response = "";
static string last_query = "";
static string current_model_name = "";

const std::string SYSTEM_PROMPT = R"(You are an expert embedded Linux and C++ engineer.
Target Environment: Ubuntu 24.04 (Kernel 6.1+), Rockchip RKNN-LLM, MPP.

Guidelines:
- Provide robust, production-ready, and modern C++ code.
- Write explanations in Czech, but keep technical terminology, code, and comments in English.
- Avoid unnecessary placeholders; provide fully functional functions/classes.
- Be concise and focus directly on technical accuracy.)";

const string LOG_DIR = "/home/orangepi/.local/share/ai-mentor/";

string escape_csv(const string& text) {
    string escaped = "";
    for (char c : text) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    return escaped;
}

void write_to_csv(const string& query, const string& response, int chunks, double seconds) {
    // Zajištění existence adresáře (C++17)
    std::filesystem::create_directories(LOG_DIR);

    time_t now = time(0);
    tm *ltm = localtime(&now);
    stringstream ss;
    ss << LOG_DIR << "ai-mentor_" << put_time(ltm, "%Y-%m-%d") << ".csv";
    string filename = ss.str();

    bool file_exists = std::filesystem::exists(filename); // Čistší než ifstream().good()
    ofstream csv(filename, ios::app);
    if (csv.is_open()) {
        if (!file_exists) {
            // Poznámka: používáš středník jako oddělovač, to je v CZ běžné, ale přípona je .csv (obvykle čárka).
            // Pokud to je záměr, nech být. Jinak změň na čárku.
            csv << "timestamp;model;chunks;seconds;speed;query;response" << endl;
        }
        double speed = (seconds > 0) ? (chunks / seconds) : 0;
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", ltm);

        csv << "\"" << time_str << "\";"
            << "\"" << current_model_name << "\";"
            << chunks << ";"
            << fixed << setprecision(2) << seconds << ";"
            << speed << ";"
            << "\"" << escape_csv(query) << "\";"
            << "\"" << escape_csv(response) << "\"" << endl;
        csv.close();
    }
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        if (result && result->text) {
            if (chunk_count == 0 && !is_pipe) cout << GREEN << BOLD << "AI Mentor: " << RESET;
            cout << result->text << flush;
            current_response += result->text;
            chunk_count++; 
        }
    } else if (state == RKLLM_RUN_FINISH) {
        // UVOLNĚNÍ PAMĚTI: Uvolníme buffer, který jsme alokovali přes strdup
        if (userdata) {
            free(userdata);
        }

        auto infer_end = Clock::now();
        double seconds = std::chrono::duration<double>(infer_end - infer_start).count();
        
        write_to_csv(last_query, current_response, chunk_count, seconds);

        if (!is_pipe) {
            // Používáme výhradně cout, nikoliv smísené printf
            cout << "\n\n" << YELLOW << "--- TELEMETRIE [" << current_model_name << "] ---" << RESET << "\n";
            cout << "Dotaz: " << total_queries 
                 << " | Chunks: " << chunk_count 
                 << " | Čas: " << fixed << setprecision(2) << seconds << " s" 
                 << " | Rychlost: " << CYAN << (seconds > 0 ? chunk_count / seconds : 0) << " chunks/s" << RESET << "\n";
            cout << YELLOW << "---------------------------------" << RESET << "\n";
        }
        current_response = "";
        chunk_count = 0;
    }
    return 0;
}

void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        if (!is_pipe) cout << YELLOW << "\nUvolňuji NPU zdroje..." << RESET << endl;
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
    exit(EXIT_FAILURE); // Čistší exit kód
}

string format_chatml_prompt(const string& system, const string& user_input, bool include_system = true) {
    string formatted = "";
    if (include_system && !system.empty()) {
        formatted += "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    formatted += "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
    return formatted;
}

int main(int argc, char **argv) {
    is_pipe = !isatty(fileno(stdin));
    if (argc < 4) {
        cerr << RED << "Použití: ./ai_mentor <model_path> <max_tokens> <context_len> [\"otázka\"]" << RESET << endl;
        return 1;
    }

    signal(SIGINT, exit_handler);

    string path_str = argv[1];
    size_t last_slash = path_str.find_last_of("/\\");
    current_model_name = (last_slash == string::npos) ? path_str : path_str.substr(last_slash + 1);

    // DOPORUČENÍ: Spouštěj fix_freq skript přes systemd service, ne odtud.
    // Zabraňuje to zasekávání programu při vyžádání hesla.
    if (!is_pipe) cout << CYAN << "Optimalizuji hardware..." << RESET << endl;
    system("sudo /home/orangepi/.local/bin/fix_freq_rk3588.sh 2>/dev/null >/dev/null");

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = (char*)path_str.c_str();
    param.max_new_tokens = atoi(argv[2]);
    param.max_context_len = atoi(argv[3]);
    param.top_k = 1; 
    param.top_p = 0.9; 
    param.temperature = 0.3;
    param.repeat_penalty = 1.1; 
    param.skip_special_token = true;
    param.extend_param.n_batch = 1; 
    param.extend_param.enabled_cpus_num = 4;
    param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7);

    if (!is_pipe) cout << CYAN << "Inicializace RKLLM..." << RESET << endl;
    int ret = rkllm_init(&llmHandle, &param, callback);

    if (ret != 0) { 
        cerr << RED << "Chyba initu RKLLM: " << ret << RESET << endl; 
        return 1; 
    }

    RKLLMInput rkllm_input{};
    RKLLMInferParam infer_param{};
    infer_param.mode = RKLLM_INFER_GENERATE;

    // Jednorázový dotaz z argumentu nebo pipe
    if (argc >= 5 || is_pipe) {
        string input_text;
        if (argc >= 5) input_text = argv[4];
        else { string line; while (getline(cin, line)) input_text += line + "\n"; }

        if (!input_text.empty()) {
            last_query = input_text;
            infer_param.keep_history = 0; 
            string full_prompt = format_chatml_prompt(SYSTEM_PROMPT, input_text, true);
            
            // BEZPEČNÁ ALOKACE: strdup vytvoří hlubokou kopii v C-heapu, která přežije zánik lokální proměnné.
            // Bude dealokována v callbacku (RKLLM_RUN_FINISH) přes parametr userdata.
            char* prompt_c_str = strdup(full_prompt.c_str());

            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.prompt_input = prompt_c_str;
            
            total_queries++;
            infer_start = Clock::now();
            // Předáme ukazatel na paměť, aby ho callback mohl uvolnit
            rkllm_run(llmHandle, &rkllm_input, &infer_param, prompt_c_str); 
        }
        rkllm_destroy(llmHandle);
        return 0;
    }

    // Interaktivní smyčka
    cout << GREEN << BOLD << "AI Mentor připraven." << RESET << endl;
    string input;
    bool is_first_turn = true;

    while (true) {
        cout << BLUE << BOLD << "\nJan: " << RESET;
        if (!getline(cin, input) || input == "exit") break;
        
        if (input == "/clear") { 
            rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr); 
            is_first_turn = true;
            cout << YELLOW << "KV Cache vyčištěna." << RESET << endl;
            continue; 
        }
        
        last_query = input;
        infer_param.keep_history = 1;
        string full_prompt = format_chatml_prompt(SYSTEM_PROMPT, input, is_first_turn);
        is_first_turn = false;

        // Opět bezpečná alokace přes strdup
        char* prompt_c_str = strdup(full_prompt.c_str());

        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = prompt_c_str;
        
        total_queries++;
        infer_start = Clock::now();
        rkllm_run(llmHandle, &rkllm_input, &infer_param, prompt_c_str);
    }

    if (llmHandle) rkllm_destroy(llmHandle);
    return 0;
}
