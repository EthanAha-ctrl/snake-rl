#include "history_stacker.h"

history_stacker::history_stacker(int obs_dim, int action_dim, int history_len) : obs_dim(obs_dim), action_dim(action_dim), history_len(history_len) {}

void history_stacker::reset(const std::vector<float> &initial_obs, float default_obs, float default_action)
{
    obs_history.clear();
    action_history.clear();

    std::vector<float> zero_obs(obs_dim, default_obs);
    std::vector<float> zero_action(action_dim, default_action);

    for (int i = 0; i < history_len - 1; ++i)
    {
        obs_history.push_back(zero_obs);
        action_history.push_back(zero_action);
    }

    // Last history is the initial current observation, and zero action
    obs_history.push_back(initial_obs);
    action_history.push_back(zero_action);
}

void history_stacker::append(const std::vector<float> &obs, const std::vector<float> &action)
{
    obs_history.push_back(obs);
    if (obs_history.size() > history_len)
    {
        obs_history.pop_front();
    }

    action_history.push_back(action);
    if (action_history.size() > history_len)
    {
        action_history.pop_front();
    }
}

std::vector<float> history_stacker::stacked() const
{
    std::vector<float> result;
    // Total dim: obs_dim * history_len + action_dim * history_len
    result.reserve(obs_dim * history_len + action_dim * history_len);

    // Concatenate all observations first
    for (const auto &obs : obs_history)
    {
        result.insert(result.end(), obs.begin(), obs.end());
    }

    // Concatenate all actions
    for (const auto &act : action_history)
    {
        result.insert(result.end(), act.begin(), act.end());
    }

    return result;
}
