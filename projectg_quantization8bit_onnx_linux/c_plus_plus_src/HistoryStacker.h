#ifndef HISTORY_STACKER_H
#define HISTORY_STACKER_H

#include <vector>
#include <deque>

class HistoryStacker {
public:
    HistoryStacker(int obs_dim, int action_dim, int history_len);

    void reset(const std::vector<float>& initial_obs, float default_obs = -1.0f, float default_action = -1.0f);
    void append(const std::vector<float>& obs, const std::vector<float>& action);
    
    // Returns a flattened array of [obs_dim*history_len + action_dim*history_len]
    std::vector<float> stacked() const;

private:
    int obs_dim_;
    int action_dim_;
    int history_len_;

    std::deque<std::vector<float>> obs_history_;
    std::deque<std::vector<float>> action_history_;
};

#endif // HISTORY_STACKER_H
