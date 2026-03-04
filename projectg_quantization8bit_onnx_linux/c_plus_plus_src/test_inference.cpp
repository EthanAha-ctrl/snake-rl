#include <iostream>
#include <string>
#include <vector>

#include "coc_env.h"
#include "engine.h"
#include "history_stacker.h"

int main(int argc, char *argv[])
{
    std::cout << "Starting C-API Test Inference..." << std::endl;

    // 1. Initialize Engine (C API)
    std::string hrnet_path = "../hrnet_int8.onnx";
    std::string transformer_path = "../transformer_int8.onnx";
    std::string actor_path = "../sac_actor.onnx";

    void *engine = engine_init(hrnet_path.c_str(), transformer_path.c_str(), actor_path.c_str());
    if (!engine)
    {
        std::cerr << "Failed to initialize engine." << std::endl;
        return 1;
    }

    // 2. Initialize Environment & History Stacker
    std::string bson_path = "../coc_images.bson";

    try
    {
        coc_env env(bson_path);
        history_stacker actual_stacker(3300, 2, 10);

        // 3. Run an Episode
        std::cout << "Starting Episode..." << std::endl;
        std::vector<float> prev_diff_res = env.reset();
        float current_diff = prev_diff_res[0];

        // Process initial obs
        auto img_data = env.get_interpolated_image(current_diff * 10.0f);
        int hrnet_out_size = 0;
        const float *hrnet_out = engine_inference(engine, MODEL_HRNET, img_data.first.data(), img_data.first.size(), &hrnet_out_size);
        if (!hrnet_out || hrnet_out_size != 3000)
            throw std::runtime_error("Initial HRNet inference failed.");

        std::vector<float> obs;
        obs.reserve(3300);
        obs.insert(obs.end(), img_data.second.begin(), img_data.second.end());
        obs.insert(obs.end(), hrnet_out, hrnet_out + 3000);

        actual_stacker.reset(obs, -1.0f, -1.0f);

        float total_reward_guess = 0.0f;
        float total_reward_trigger = 0.0f;
        bool done = false;

        while (!done)
        {
            // A. Stack History
            std::vector<float> stacked_obs = actual_stacker.stacked();

            // B. Transformer Inference
            int tr_out_size = 0;
            const float *tr_out_arr = engine_inference(engine, MODEL_TRANSFORMER, stacked_obs.data(), stacked_obs.size(), &tr_out_size);
            if (!tr_out_arr)
                throw std::runtime_error("Transformer inference failed.");

            // C. Actor Inference
            int act_out_size = 0;
            const float *act_out_arr = engine_inference(engine, MODEL_ACTOR, tr_out_arr, tr_out_size, &act_out_size);
            if (!act_out_arr || act_out_size < 2)
                throw std::runtime_error("Actor inference failed.");

            std::vector<float> action(act_out_arr, act_out_arr + 2); // [guess, trigger]

            // D. Environment Step
            env_step_result step_res = env.step(action);
            current_diff = step_res.obs[0];

            // Recompute next obs via HRNet
            auto next_img_data = env.get_interpolated_image(current_diff * 10.0f);
            int next_hrnet_out_size = 0;
            const float *next_hrnet_out = engine_inference(engine, MODEL_HRNET, next_img_data.first.data(), next_img_data.first.size(), &next_hrnet_out_size);
            if (!next_hrnet_out || next_hrnet_out_size != 3000)
                throw std::runtime_error("HRNet inference failed in step.");

            std::vector<float> next_obs;
            next_obs.reserve(3300);
            next_obs.insert(next_obs.end(), next_img_data.second.begin(), next_img_data.second.end());
            next_obs.insert(next_obs.end(), next_hrnet_out, next_hrnet_out + 3000);

            actual_stacker.append(next_obs, action);

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
        engine_free(engine);
        return 1;
    }

    engine_free(engine);
    return 0;
}
