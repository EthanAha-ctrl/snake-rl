#include "CoCEnv.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>


// Dependency assumes nlohmann/json for BSON parsing and stb_image.h for PNG
// decoding
#include "stb_image.h"
#include <nlohmann/json.hpp>


// Assume ONNX Runtime is included by the user
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;

CoCEnv::CoCEnv(const std::string &bson_file_path, Ort::Session *hrnet_session)
    : hrnet_session_(hrnet_session), rng_(std::random_device{}()) {
  init_data(bson_file_path);
}

CoCEnv::~CoCEnv() {}

void CoCEnv::init_data(const std::string &bson_file_path) {
  std::ifstream f(bson_file_path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open BSON file: " + bson_file_path);
  }

  std::vector<uint8_t> bson_data((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
  int offset = 0;
  int count = 0;

  while (offset < bson_data.size()) {
    int32_t size;
    std::memcpy(&size, &bson_data[offset], sizeof(int32_t));

    std::vector<uint8_t> single_doc(bson_data.begin() + offset,
                                    bson_data.begin() + offset + size);
    json j = json::from_bson(single_doc);

    std::string key = j["key"];
    int label_r = j["label_r"];

    // nlohmann::json stores binary data as std::vector<uint8_t>
    std::vector<uint8_t> image_png = j["image_png"].get_binary();

    label_to_keys_[label_r].push_back(key);
    key_to_image_png_[key] = image_png;

    offset += size;
    count++;
  }

  total_entries_ = count;
  std::cout << "Loaded " << count << " BSON documents into CoCEnv."
            << std::endl;
}

std::vector<uint8_t> CoCEnv::get_random_image_for_label(int label) {
  auto &keys = label_to_keys_[label];
  if (keys.empty())
    return {};

  int idx = current_img_index_ % keys.size();
  std::string key = keys[idx];
  return key_to_image_png_[key];
}

std::vector<float> CoCEnv::decode_png(const std::vector<uint8_t> &png_bytes) {
  int x, y, channels;
  // Force grayscale (1 channel)
  uint8_t *data = stbi_load_from_memory(png_bytes.data(), png_bytes.size(), &x,
                                        &y, &channels, 1);
  if (!data) {
    throw std::runtime_error("Failed to decode PNG in memory.");
  }

  // We expect 480x640 size
  std::vector<float> img_float(x * y);
  for (int i = 0; i < x * y; ++i) {
    img_float[i] = static_cast<float>(data[i]) / 255.0f; // Normalize 0-1
  }

  stbi_image_free(data);
  return img_float;
}

std::vector<float> CoCEnv::get_interpolated_image(float val) {
  val = std::max(0.0f, std::min(val, 9.999f));
  int label_floor = static_cast<int>(std::floor(val));
  int label_ceil = label_floor + 1;

  float weight_ceil = val - static_cast<float>(label_floor);
  float weight_floor = 1.0f - weight_ceil;

  label_floor = std::max(0, std::min(label_floor, 9));
  label_ceil = std::max(0, std::min(label_ceil, 9));

  auto png_floor = get_random_image_for_label(label_floor);
  auto png_ceil = get_random_image_for_label(label_ceil);

  auto img_floor = decode_png(png_floor);
  auto img_ceil = decode_png(png_ceil);

  std::vector<float> mixed(img_floor.size());
  for (size_t i = 0; i < mixed.size(); ++i) {
    mixed[i] = img_floor[i] * weight_floor + img_ceil[i] * weight_ceil;
  }
  return mixed;
}

float CoCEnv::compute_expected_radius(float guess) {
  float val = guess * 10.0f;
  auto t_mix = get_interpolated_image(val); // Returns 1D vector (size 480*640)

  // 1. Run HRNet Inference
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> input_shape = {1, 1, 480, 640};
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info, t_mix.data(), t_mix.size(),
                                      input_shape.data(), input_shape.size());

  const char *input_names[] = {"image_input"};
  const char *output_names[] = {"vision_features"};

  auto output_tensors = hrnet_session_->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  float *out_arr =
      output_tensors[0]
          .GetTensorMutableData<float>(); // Shape is [1, 10, 15, 20]

  // 2. Average Pool Output (Mean on spatial dims 15x20 = 300)
  std::vector<float> logits_avg(10, 0.0f);
  int spatial_size = 15 * 20;
  for (int c = 0; c < 10; ++c) {
    float sum = 0.0f;
    for (int s = 0; s < spatial_size; ++s) {
      sum += out_arr[c * spatial_size + s];
    }
    logits_avg[c] = sum / spatial_size;
  }

  // 3. Softmax
  float max_val = *std::max_element(logits_avg.begin(), logits_avg.end());
  float sum_exp = 0.0f;
  std::vector<float> probs(10);
  for (int i = 0; i < 10; ++i) {
    probs[i] = std::exp(logits_avg[i] - max_val);
    sum_exp += probs[i];
  }
  for (int i = 0; i < 10; ++i) {
    probs[i] /= sum_exp;
  }

  // 4. Top-2 Masking and Expectation
  int top1_idx = -1, top2_idx = -1;
  float top1_val = -1.0f, top2_val = -1.0f;
  for (int i = 0; i < 10; ++i) {
    if (probs[i] > top1_val) {
      top2_val = top1_val;
      top2_idx = top1_idx;
      top1_val = probs[i];
      top1_idx = i;
    } else if (probs[i] > top2_val) {
      top2_val = probs[i];
      top2_idx = i;
    }
  }

  float norm_sum = top1_val + top2_val + 1e-9f;
  float expected_radius =
      (top1_val * top1_idx + top2_val * top2_idx) / norm_sum;

  return expected_radius;
}

std::vector<float> CoCEnv::reset() {
  std::uniform_int_distribution<int> int_dist(0, total_entries_);
  std::uniform_real_distribution<float> float_dist(min_val_, max_val_);

  current_img_index_ = int_dist(rng_);
  ground_truth_ = float_dist(rng_);
  current_step_ = 0;
  target_step_ = float_dist(rng_);
  prev_diff_ = std::abs(target_step_ - ground_truth_);
  reached_ = false;
  is_first_trial_ = true;

  diff_threshold_ = (max_val_ - min_val_) / max_steps_;

  float obs_val = std::max(
      0.0f, std::min(1.0f, compute_expected_radius(prev_diff_) / 10.0f));
  return {obs_val};
}

EnvStepResult CoCEnv::step(const std::vector<float> &action) {
  float guess_act = action[0];
  bool trigger_act = action[1] > 0.5f;

  if (trigger_act) {
    target_step_ = guess_act;
  } else {
    guess_act = target_step_;
  }

  guess_act = std::max(0.0f, std::min(1.0f, guess_act));
  float absolute_diff = std::abs(guess_act - ground_truth_);
  absolute_diff = compute_expected_radius(absolute_diff) / 10.0f;

  current_step_++;

  float r_guess = 0.0f;
  bool terminated = false;

  float improvement = prev_diff_ - absolute_diff;
  r_guess = improvement >= 0.0f ? 1.0f : -1.0f;
  r_guess -= (current_step_ * current_step_) / 10.0f;
  float r_trigger = 0.0f;

  bool reached_now = false;
  if (absolute_diff < diff_threshold_) {
    reached_now = true;
    if (!reached_)
      r_guess += 10.0f;
    else
      r_guess = 0.0f;
  }

  if (current_step_ >= max_steps_) {
    terminated = true;
    r_guess -= 10.0f;
    r_trigger -= 10.0f;
  }

  if (!reached_now) {
    if (!trigger_act)
      r_trigger -= (current_step_ * current_step_);
    else
      r_trigger += 1.0f;
  } else {
    if (!reached_) {
      if (!trigger_act)
        r_trigger -= (current_step_ * current_step_);
      else
        r_trigger += 1.0f;
    }
  }

  if (reached_) {
    if (!trigger_act) {
      r_trigger += 1.0f;
      terminated = true;
    } else {
      r_trigger = -10.0f;
    }
  }

  if (is_first_trial_) {
    is_first_trial_ = false;
    r_guess = 0.1f;
    if (reached_now && trigger_act)
      r_guess = 10.0f;
  }

  prev_diff_ = absolute_diff;
  if (absolute_diff < diff_threshold_)
    reached_ = true;

  float next_obs = std::max(0.0f, std::min(1.0f, prev_diff_));
  return {{next_obs}, {r_guess, r_trigger}, terminated, false};
}
