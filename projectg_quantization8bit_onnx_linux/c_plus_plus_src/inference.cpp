#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "coc_env.h"
#include "history_stacker.h"

int main(int argc, char *argv[])
{
    std::cout << "Starting SAC CoC Inference in pure C++..." << std::endl;

    // 1. Initialize ONNX Runtime
    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "SnakeRL_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 2. Load Models (Assuming we use INT8 models)
    std::string hrnet_path = "../hrnet_int8.onnx";
    std::string transformer_path = "../transformer_int8.onnx";
    std::string actor_path = "../sac_actor.onnx";

    // Fallbacks to unquantized if not available in your build setup
    // (user should swap names if they want FP32 tests)
    Ort::Session hrnet_session(ort_env, hrnet_path.c_str(), session_options);
    Ort::Session transformer_session(ort_env, transformer_path.c_str(), session_options);
    Ort::Session actor_session(ort_env, actor_path.c_str(), session_options);

    std::cout << "ONNX Models loaded successfully." << std::endl;

    // 3. Initialize Environment & History Stacker
    std::string bson_path = "../coc_images.bson";

    try
    {
        coc_env env(bson_path, &hrnet_session);
        history_stacker actual_stacker(1, 2, 10);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // 4. Run an Episode
        std::cout << "Starting Episode..." << std::endl;
        std::vector<float> obs = env.reset();
        actual_stacker.reset(obs, -1.0f, -1.0f);

        float total_reward_guess = 0.0f;
        float total_reward_trigger = 0.0f;
        bool done = false;

        while (!done)
        {
            // A. Stack History
            std::vector<float> stacked_obs = actual_stacker.stacked();

            // B. Transformer Inference
            std::vector<int64_t> tr_input_shape = {1, static_cast<int64_t>(stacked_obs.size())};
            Ort::Value tr_input_tensor = Ort::Value::CreateTensor<float>(memory_info, stacked_obs.data(), stacked_obs.size(), tr_input_shape.data(), tr_input_shape.size());

            const char *tr_input_names[] = {"obs_stack"};
            const char *tr_output_names[] = {"encoded_features"};

            auto tr_output_tensors = transformer_session.Run(Ort::RunOptions{nullptr}, tr_input_names, &tr_input_tensor, 1, tr_output_names, 1);

            float *tr_out_ptr = tr_output_tensors[0].GetTensorMutableData<float>();
            size_t tr_out_size = 52; // Transformer output dim

            // C. Actor Inference
            std::vector<int64_t> act_input_shape = {1, static_cast<int64_t>(tr_out_size)};
            Ort::Value act_input_tensor = Ort::Value::CreateTensor<float>(memory_info, tr_out_ptr, tr_out_size, act_input_shape.data(), act_input_shape.size());

            const char *act_input_names[] = {"features"};
            const char *act_output_names[] = {"action"};

            auto act_output_tensors = actor_session.Run(Ort::RunOptions{nullptr}, act_input_names, &act_input_tensor, 1, act_output_names, 1);

            float *action_ptr = act_output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> action(action_ptr, action_ptr + 2); // [guess, trigger]

            // D. Environment Step
            env_step_result step_res = env.step(action);

            actual_stacker.append(step_res.obs, action);

            total_reward_guess += step_res.reward[0];
            total_reward_trigger += step_res.reward[1];
            done = step_res.terminated || step_res.truncated;

            std::cout << "  Guess: " << action[0] << " | Trigger: " << action[1] << " | Reward: [" << step_res.reward[0] << ", " << step_res.reward[1] << "]" << std::endl;
        }

        std::cout << "Episode Finished!" << std::endl;
        std::cout << "Total Reward (Guess): " << total_reward_guess << std::endl;
        std::cout << "Total Reward (Trigger): " << total_reward_trigger << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
