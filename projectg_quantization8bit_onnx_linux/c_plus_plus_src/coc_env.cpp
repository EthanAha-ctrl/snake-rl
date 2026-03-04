#include "coc_env.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Dependency assumes nlohmann/json for BSON parsing and stb_image.h for PNG
// decoding
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <nlohmann/json.hpp>

// Assume ONNX Runtime is included by the user
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;

coc_env::coc_env(const std::string &bson_file_path, Ort::Session *hrnet_session) : hrnet_session(hrnet_session), rng(std::random_device{}()) { init_data(bson_file_path); }

coc_env::~coc_env() {}

void coc_env::init_data(const std::string &bson_file_path)
{
    std::ifstream f(bson_file_path, std::ios::binary);
    if (!f.is_open())
    {
        throw std::runtime_error("Could not open BSON file: " + bson_file_path);
    }

    std::vector<uint8_t> bson_data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int offset = 0;
    int count = 0;

    while (offset < bson_data.size())
    {
        int32_t size;
        std::memcpy(&size, &bson_data[offset], sizeof(int32_t));

        std::vector<uint8_t> single_doc(bson_data.begin() + offset, bson_data.begin() + offset + size);
        json j = json::from_bson(single_doc);

        std::string key = j["key"];
        int label_r = j["label_r"];

        // nlohmann::json stores binary data as std::vector<uint8_t>
        std::vector<uint8_t> image_png = j["image_png"].get_binary();

        label_to_keys[label_r].push_back(key);
        key_to_image_png[key] = image_png;

        if (j.contains("sharpness_grid"))
        {
            std::vector<uint8_t> sharp_bytes = j["sharpness_grid"].get_binary();
            // Expected 15 * 20 floats = 300 * 4 = 1200 bytes
            std::vector<float> sharp_floats(300);
            std::memcpy(sharp_floats.data(), sharp_bytes.data(), 1200);
            key_to_sharpness[key] = sharp_floats;
        }

        offset += size;
        count++;
    }

    total_entries = count;
    std::cout << "Loaded " << count << " BSON documents into coc_env." << std::endl;
}

std::pair<std::vector<uint8_t>, std::vector<float>> coc_env::get_random_data_for_label(int label)
{
    auto &keys = label_to_keys[label];
    if (keys.empty())
        return {{}, std::vector<float>(300, 0.0f)};

    int idx = current_img_index % keys.size();
    std::string key = keys[idx];
    
    std::vector<float> sharp(300, 0.0f);
    if (key_to_sharpness.find(key) != key_to_sharpness.end()) {
        sharp = key_to_sharpness[key];
    }
    
    return {key_to_image_png[key], sharp};
}

std::vector<float> coc_env::decode_png(const std::vector<uint8_t> &png_bytes)
{
    if (png_bytes.empty()) return std::vector<float>(480 * 640, 0.0f);

    int x, y, channels;
    // Force grayscale (1 channel)
    uint8_t *data = stbi_load_from_memory(png_bytes.data(), png_bytes.size(), &x, &y, &channels, 1);
    if (!data)
    {
        throw std::runtime_error("Failed to decode PNG in memory.");
    }

    // We expect 480x640 size
    std::vector<float> img_float(x * y);
    for (int i = 0; i < x * y; ++i)
    {
        img_float[i] = static_cast<float>(data[i]) / 255.0f; // Normalize 0-1
    }

    stbi_image_free(data);
    return img_float;
}

std::vector<float> coc_env::get_interpolated_obs(float val, float &out_expected_radius)
{
    val = std::max(0.0f, std::min(val, 9.999f));
    int label_floor = static_cast<int>(std::floor(val));
    int label_ceil = label_floor + 1;

    float weight_ceil = val - static_cast<float>(label_floor);
    float weight_floor = 1.0f - weight_ceil;

    label_floor = std::max(0, std::min(label_floor, 9));
    label_ceil = std::max(0, std::min(label_ceil, 9));

    auto data_floor = get_random_data_for_label(label_floor);
    auto data_ceil = get_random_data_for_label(label_ceil);

    auto img_floor = decode_png(data_floor.first);
    auto img_ceil = decode_png(data_ceil.first);

    std::vector<float> t_mix(img_floor.size());
    for (size_t i = 0; i < t_mix.size(); ++i)
    {
        t_mix[i] = img_floor[i] * weight_floor + img_ceil[i] * weight_ceil;
    }

    // 1. Run HRNet Inference
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 1, 480, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, t_mix.data(), t_mix.size(), input_shape.data(), input_shape.size());

    const char *input_names[] = {"image_input"};
    const char *output_names[] = {"vision_features"};

    auto output_tensors = hrnet_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float *out_arr = output_tensors[0].GetTensorMutableData<float>(); // Shape is [1, 10, 15, 20]

    // 2. Average Pool Output (Mean on spatial dims 15x20 = 300)
    std::vector<float> logits_avg(10, 0.0f);
    int spatial_size = 15 * 20;
    for (int c = 0; c < 10; ++c)
    {
        float sum = 0.0f;
        for (int s = 0; s < spatial_size; ++s)
        {
            sum += out_arr[c * spatial_size + s];
        }
        logits_avg[c] = sum / spatial_size;
    }

    // 3. Softmax
    float max_val_logit = *std::max_element(logits_avg.begin(), logits_avg.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(10);
    for (int i = 0; i < 10; ++i)
    {
        probs[i] = std::exp(logits_avg[i] - max_val_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < 10; ++i)
    {
        probs[i] /= sum_exp;
    }

    // 4. Top-2 Masking and Expectation
    int top1_idx = -1, top2_idx = -1;
    float top1_val = -1.0f, top2_val = -1.0f;
    for (int i = 0; i < 10; ++i)
    {
        if (probs[i] > top1_val)
        {
            top2_val = top1_val;
            top2_idx = top1_idx;
            top1_val = probs[i];
            top1_idx = i;
        }
        else if (probs[i] > top2_val)
        {
            top2_val = probs[i];
            top2_idx = i;
        }
    }

    float norm_sum = top1_val + top2_val + 1e-9f;
    out_expected_radius = (top1_val * top1_idx + top2_val * top2_idx) / norm_sum;

    // 5. Construct 3300-dim Obs
    std::vector<float> obs_tensor;
    obs_tensor.reserve(3300);

    // Sharpness (1x15x20 = 300)
    auto &s_floor = data_floor.second;
    auto &s_ceil = data_ceil.second;
    for (size_t i = 0; i < 300; ++i)
    {
        float s_mix = s_floor[i] * weight_floor + s_ceil[i] * weight_ceil;
        float sharpness_obs = (s_mix / 640.0f / 480.0f) * sharpness_scale;
        obs_tensor.push_back(sharpness_obs);
    }

    // Vision Features (10x15x20 = 3000)
    for (size_t i = 0; i < 3000; ++i)
    {
        obs_tensor.push_back(out_arr[i]);
    }

    return obs_tensor;
}

std::vector<float> coc_env::reset()
{
    std::uniform_int_distribution<int> int_dist(0, total_entries);
    std::uniform_real_distribution<float> float_dist(min_val, max_val);
    std::uniform_real_distribution<float> sharp_dist(0.5f, 1.5f);

    current_img_index = int_dist(rng);
    sharpness_scale = sharp_dist(rng);
    ground_truth = float_dist(rng);
    current_step = 0;
    target_step = float_dist(rng);
    prev_diff = std::abs(target_step - ground_truth);
    prev_position = target_step;
    reached = false;
    is_first_trial = true;

    fsm = "coarse search";
    fsm_overshoot_count = 0;

    diff_threshold = (max_val - min_val) / max_steps;

    float expected_radius = 0.0f;
    std::vector<float> obs = get_interpolated_obs(prev_diff * 10.0f, expected_radius);
    return obs;
}

env_step_result coc_env::step(const std::vector<float> &action)
{
    float guess_act = action[0];
    bool trigger_act = action[1] > 0.5f;

    if (trigger_act)
    {
        target_step = guess_act;
    }
    else
    {
        guess_act = target_step;
    }

    guess_act = std::max(0.0f, std::min(1.0f, guess_act));
    float absolute_diff = std::abs(guess_act - ground_truth);
    
    float expected_radius = 0.0f;
    std::vector<float> obs = get_interpolated_obs(absolute_diff * 10.0f, expected_radius);

    current_step++;

    float r_guess = 0.0f;
    float r_trigger = 0.0f;
    float improvement = prev_diff - absolute_diff;

    if (is_first_trial)
    {
        is_first_trial = false;
        r_guess = 0.1f;
        r_trigger = trigger_act ? 0.1f : -10.0f;
    }
    else
    {
        float sign_prev = (prev_position - ground_truth >= 0.0f) ? 1.0f : -1.0f;
        float sign_curr = (guess_act - ground_truth >= 0.0f) ? 1.0f : -1.0f;
        float sign = sign_prev * sign_curr;
        
        if (sign < 0)
        {
            fsm_overshoot_count++;
        }
        
        if (fsm == "coarse search")
        {
            if (fsm_overshoot_count == 1)
            {
                fsm = "fine search";
            }
            else
            {
                fsm = "coarse search";
            }

            if (trigger_act)
            {
                r_trigger = 0.1f;
            }
            else
            {
                r_trigger = -0.1f;
            }

            if (sign < 0)
            {
                r_guess = 1.0f;
            }
            else
            {
                r_guess = -absolute_diff;
            }
        }
        else
        {
            // fine search
            r_guess += -(current_step * current_step) / 10.0f;
            r_guess += improvement * 10.0f;
            
            if (!trigger_act && prev_diff < diff_threshold)
            {
                r_trigger = 2.0f;
            }
            else if (trigger_act && prev_diff < diff_threshold)
            {
                r_trigger = -2.0f;
            }
            else if (!trigger_act && prev_diff > diff_threshold)
            {
                r_trigger = -2.0f;
            }
            else if (trigger_act && prev_diff > diff_threshold)
            {
                r_trigger = 1.0f;
            }
        }
    }

    bool terminated = false;
    if (!is_first_trial && absolute_diff < diff_threshold && !trigger_act)
    {
        terminated = true;
        r_guess = 10.0f;
        r_trigger = 10.0f;
    }

    if (current_step >= max_steps)
    {
        terminated = true;
        r_guess = -10.0f;
        r_trigger = -10.0f;
    }

    prev_diff = absolute_diff;
    prev_position = guess_act;

    return {obs, {r_guess, r_trigger}, terminated, false};
}
