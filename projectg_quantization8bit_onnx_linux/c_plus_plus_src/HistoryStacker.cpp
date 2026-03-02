#include "HistoryStacker.h"

HistoryStacker::HistoryStacker(int obs_dim, int action_dim, int history_len)
    : obs_dim_(obs_dim), action_dim_(action_dim), history_len_(history_len) {
}

void HistoryStacker::reset(const std::vector<float>& initial_obs, float default_obs, float default_action) {
    obs_history_.clear();
    action_history_.clear();

    std::vector<float> zero_obs(obs_dim_, default_obs);
    std::vector<float> zero_action(action_dim_, default_action);

    for (int i = 0; i < history_len_ - 1; ++i) {
        obs_history_.push_back(zero_obs);
        action_history_.push_back(zero_action);
    }
    
    // Last history is the initial current observation, and zero action
    obs_history_.push_back(initial_obs);
    action_history_.push_back(zero_action);
}

void HistoryStacker::append(const std::vector<float>& obs, const std::vector<float>& action) {
    obs_history_.push_back(obs);
    if (obs_history_.size() > history_len_) {
        obs_history_.pop_front();
    }

    action_history_.push_back(action);
    if (action_history_.size() > history_len_) {
        action_history_.pop_front();
    }
}

std::vector<float> HistoryStacker::stacked() const {
    std::vector<float> result;
    // Total dim: obs_dim * history_len + action_dim * history_len
    result.reserve(obs_dim_ * history_len_ + action_dim_ * history_len_);

    // Concatenate all observations first
    for (const auto& obs : obs_history_) {
        result.insert(result.end(), obs.begin(), obs.end());
    }

    // Concatenate all actions
    for (const auto& act : action_history_) {
        result.insert(result.end(), act.begin(), act.end());
    }

    return result;
}
