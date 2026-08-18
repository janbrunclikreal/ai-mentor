#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include "rkllm.h"

int test_cb(RKLLMResult *result, void *userdata, LLMCallState state) {
    std::cout << "[CB State=" << state << "] " << std::flush;
    if (state == RKLLM_RUN_NORMAL) {
        if (result && result->text) {
            std::cout << "TEXT: " << result->text << std::flush;
        }
    } else if (state == RKLLM_RUN_WAITING) {
        std::cout << "WAIT" << std::flush;
    } else if (state == RKLLM_RUN_FINISH) {
        std::cout << " -> FINISH" << std::endl;
    } else if (state == RKLLM_RUN_ERROR) {
        std::cerr << " -> ERROR" << std::endl;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Pouziti: ./diag <cesta_k_souboru_rkllm>" << std::endl;
        return 1;
    }

    LLMHandle handle = nullptr;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.max_context_len = 2048;
    param.max_new_tokens = 64;
    param.skip_special_token = true;

    std::cout << "Inicializace RKLLM s modelem: " << argv[1] << std::endl;
    int ret = rkllm_init(&handle, &param, test_cb);
    if (ret != 0) {
        std::cerr << "Chyba rkllm_init: " << ret << std::endl;
        return 1;
    }

    std::cout << "Spoustim inferenci..." << std::endl;
    RKLLMInput input;
    std::memset(&input, 0, sizeof(RKLLMInput));
    input.role = nullptr;
    input.enable_thinking = false;
    input.input_type = RKLLM_INPUT_PROMPT;
    input.prompt_input = (char*)"Ahoj";

    RKLLMInferParam infer_param;
    std::memset(&infer_param, 0, sizeof(RKLLMInferParam));
    infer_param.mode = RKLLM_INFER_GENERATE;

    int run_ret = rkllm_run(handle, &input, &infer_param, nullptr);
    std::cout << "\nrkllm_run navrat: " << run_ret << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(2));
    rkllm_destroy(handle);
    return 0;
}
