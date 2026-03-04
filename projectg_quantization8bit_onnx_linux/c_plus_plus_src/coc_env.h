#ifndef COC_ENV_H
#define COC_ENV_H

#include <map>
#include <random>
#include <string>
#include <vector>

#include "engine.h"

struct env_step_result
{
    std::vector<float> obs;
    std::vector<float> reward; // [r_guess, r_trigger]
    bool terminated;
    bool truncated;
};

class coc_env
{
  public:
    coc_env(const std::string &bson_file_path, EngineHandle engine);
    ~coc_env();

    std::vector<float> reset();
    env_step_result step(const std::vector<float> &action);
    std::pair<std::vector<float>, std::vector<float>> get_interpolated_image(float val);

  private:
    float min_val = 0.0f;
    float max_val = 1.0f;
    int max_steps = 10;

    // State
    float ground_truth;
    int current_step;
    float target_step;
    float diff_threshold;
    bool reached;
    bool is_first_trial;
    float prev_diff;
    float prev_position;
    int current_img_index;
    float sharpness_scale;

    std::string fsm;
    int fsm_overshoot_count;

    // Data Storage
    int total_entries;
    std::map<int, std::vector<std::string>> label_to_keys;
    std::map<std::string, std::vector<uint8_t>> key_to_image_png; // Store raw PNG bytes
    std::map<std::string, std::vector<float>> key_to_sharpness;   // 15x20 floats

    std::mt19937 rng;

    // Initialization
    void init_data(const std::string &bson_file_path);

    // Core Logic
    std::pair<std::vector<uint8_t>, std::vector<float>> get_random_data_for_label(int label);

    // Helpers
    std::vector<float> decode_png(const std::vector<uint8_t> &png_bytes); // Returns flat float32 [H*W]
};

#endif // COC_ENV_H
