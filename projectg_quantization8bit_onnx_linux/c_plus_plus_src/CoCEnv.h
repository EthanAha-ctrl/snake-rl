#ifndef COC_ENV_H
#define COC_ENV_H

#include <map>
#include <random>
#include <string>
#include <vector>


// Forward declaration for ONNX Runtime to keep header clean if possible
// Alternatively, assume caller includes ONNX Runtime headers.
namespace Ort {
class Session;
class Env;
} // namespace Ort

struct EnvStepResult {
  std::vector<float> obs;
  std::vector<float> reward; // [r_guess, r_trigger]
  bool terminated;
  bool truncated;
};

class CoCEnv {
public:
  CoCEnv(const std::string &bson_file_path, Ort::Session *hrnet_session);
  ~CoCEnv();

  std::vector<float> reset();
  EnvStepResult step(const std::vector<float> &action);

private:
  float min_val_ = 0.0f;
  float max_val_ = 1.0f;
  int max_steps_ = 10;

  // State
  float ground_truth_;
  int current_step_;
  float target_step_;
  float diff_threshold_;
  bool reached_;
  bool is_first_trial_;
  float prev_diff_;
  int current_img_index_;

  // Data Storage
  int total_entries_;
  std::map<int, std::vector<std::string>> label_to_keys_;
  std::map<std::string, std::vector<uint8_t>>
      key_to_image_png_; // Store raw PNG bytes

  Ort::Session *hrnet_session_;
  std::mt19937 rng_;

  // Initialization
  void init_data(const std::string &bson_file_path);

  // Core Logic
  std::vector<uint8_t> get_random_image_for_label(int label);
  std::vector<float> get_interpolated_image(float val);
  float compute_expected_radius(float guess);

  // Helpers
  std::vector<float> decode_png(
      const std::vector<uint8_t> &png_bytes); // Returns flat float32 [H*W]
};

#endif // COC_ENV_H
