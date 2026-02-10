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
static int token_count = 0;
static int total_queries = 0;
static bool is_pipe = false;
static string current_response = "";
static string last_query = "";
static string current_model_name = "";

const string SYSTEM_PROMPT = "Jsi technický mentor. Tvůj žák je Jan (muž). Oslovuj ho výhradně jménem Jan nebo Jane. Mluv vždy česky, věcně a technicky přesně. ";
const string LOG_DIR = "/home/orangepi/.local/share/ai-mentor/";

// Funkce pro bezpečné ošetření textu do CSV (zdvojení uvozovek)
string escape_csv(string text) {
    string escaped = "";
    for (char c : text) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    return escaped;
}

// Funkce pro zápis do CSV s denní rotací názvu souboru
void write_to_csv(const string& query, const string& response, int tokens, double seconds) {
    // Generování názvu souboru podle aktuálního data
    time_t now = time(0);
    tm *ltm = localtime(&now);
    stringstream ss;
    ss << LOG_DIR << "ai-mentor_" << put_time(ltm, "%Y-%m-%d") << ".csv";
    string filename = ss.str();

    // Kontrola existence souboru pro zápis hlavičky
    bool file_exists = ifstream(filename).good();
    
    ofstream csv(filename, ios::app);
    if (csv.is_open()) {
        if (!file_exists) {
            // Hlavička pro MySQL import
            csv << "timestamp;model;tokens;seconds;speed;query;response" << endl;
        }

        double speed = (seconds > 0) ? (tokens / seconds) : 0;
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", ltm);

        // Zápis řádku: "hodnota";"hodnota"
        csv << "\"" << time_str << "\";";
        csv << "\"" << current_model_name << "\";";
        csv << tokens << ";";
        csv << fixed << setprecision(2) << seconds << ";";
        csv << speed << ";";
        csv << "\"" << escape_csv(query) << "\";";
        csv << "\"" << escape_csv(response) << "\"" << endl;
        
        csv.close();
    }
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        if (token_count == 0 && !is_pipe) cout << GREEN << BOLD << "AI Mentor: " << RESET;
        printf("%s", result->text);
        current_response += result->text;
        fflush(stdout);
        token_count++; 
    } else if (state == RKLLM_RUN_FINISH) {
        auto infer_end = Clock::now();
        double seconds = std::chrono::duration<double>(infer_end - infer_start).count();
        
        write_to_csv(last_query, current_response, token_count, seconds);

        if (!is_pipe) {
            printf("\n\n%s--- TELEMETRIE [%s] ---%s\n", YELLOW, current_model_name.c_str(), RESET);
            printf("Dotaz: %d | Tokeny: %d | Čas: %.2f s | Rychlost: %s%.2f tok/s%s\n", 
                   total_queries, token_count, seconds, CYAN, (token_count / seconds), RESET);
            printf("%s------------------%s\n", YELLOW, RESET);
        }
        current_response = "";
        token_count = 0;
    }
    return 0;
}

void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        if (!is_pipe) cout << YELLOW << "\nUvolňuji NPU zdroje..." << RESET << endl;
        rkllm_destroy(llmHandle);
    }
    exit(signal);
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

    if (!is_pipe) cout << CYAN << "Optimalizuji hardware..." << RESET << endl;
    system("sudo /home/orangepi/.local/bin/fix_freq_rk3588.sh 2>/dev/null >/dev/null");

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = (char*)path_str.c_str();
    param.max_new_tokens = atoi(argv[2]);
    param.max_context_len = atoi(argv[3]);
    param.top_k = 1; param.top_p = 0.9; param.temperature = 0.3;
    param.repeat_penalty = 1.1; param.skip_special_token = true;
    param.extend_param.n_batch = 1; param.extend_param.enabled_cpus_num = 4;
    param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7);

    int saved_stdout = -1;
    if (is_pipe) {
        fflush(stdout);
        saved_stdout = dup(STDOUT_FILENO);
        int dev_null = open("/dev/null", O_WRONLY);
        dup2(dev_null, STDOUT_FILENO);
        close(dev_null);
    }

    if (!is_pipe) cout << CYAN << "Inicializace RKLLM..." << RESET << endl;
    int ret = rkllm_init(&llmHandle, &param, callback);

    if (is_pipe) {
        fflush(stdout);
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }

    if (ret != 0) { cerr << RED << "Chyba initu: " << ret << RESET << endl; return 1; }

    RKLLMInput rkllm_input{};
    RKLLMInferParam infer_param{};
    infer_param.mode = RKLLM_INFER_GENERATE;
    infer_param.keep_history = 1;

    if (argc >= 5 || is_pipe) {
        string input_text;
        if (argc >= 5) input_text = argv[4];
        else { string line; while (getline(cin, line)) input_text += line + "\n"; }

        if (!input_text.empty()) {
            last_query = input_text;
            string prompt = SYSTEM_PROMPT + input_text;
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.prompt_input = (char*)prompt.c_str();
            total_queries++;
            infer_start = Clock::now();
            rkllm_run(llmHandle, &rkllm_input, &infer_param, nullptr);
        }
        rkllm_destroy(llmHandle);
        return 0;
    }

    cout << GREEN << BOLD << "AI Mentor připraven." << RESET << endl;
    string input;
    while (true) {
        cout << BLUE << BOLD << "\nJan: " << RESET;
        if (!getline(cin, input) || input == "exit") break;
        if (input == "/clear") { rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr); continue; }
        last_query = input;
        string prompt = SYSTEM_PROMPT + input;
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = (char*)prompt.c_str();
        total_queries++;
        infer_start = Clock::now();
        rkllm_run(llmHandle, &rkllm_input, &infer_param, nullptr);
    }
    rkllm_destroy(llmHandle);
    return 0;
}
