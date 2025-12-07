# snake-rl

## Project 1 cartpole:

basic cartpole with SB3

## Project 2 cartpole:

image feature extraction: from single image, extract the position and angle
performance:
	
```
97% position error within: -0.0170 to 0.0204
97% angle error (deg) within: -1.3026 to 1.6789
```

## Project 3 cartpole:
basic cartpole with sequence of position state vector.

	
	------------------------------------------
	| rollout/                |             |
	|    ep_len_mean          | 288         |
	|    ep_rew_mean          | 288         |
	| time/                   |             |
	|    fps                  | 1151        |
	|    iterations           | 98          |
	|    time_elapsed         | 173         |
	|    total_timesteps      | 200000      |
	| train/                  |             |
	|    approx_kl            | 0.00108728  |
	|    clip_fraction        | 0.0972      |
	|    clip_range           | 0.20        |
	|    entropy_loss         | -0.53       |
	|    explained_variance   | 0.785       |
	|    learning_rate        | 0.000300    |
	|    loss                 | 83.3147     |
	|    n_updates            | 7820        |
	|    policy_gradient_loss | 0.0101      |
	|    value_loss           | 83.3099     |
	------------------------------------------


## Project 4 cartpole:
basic cartpole with sequence of image-extracted state vector

## Project 4.5 cartpole: Project 4
custom environment

## Project 5 CoC:
basic CoC with sequence of position state vector, supervised learning

## Project 6 CoC: Project 3
basic CoC with sequence of position state vector, RL

## Project 7 CoC:
feature extractor, supervised learning

## Project 8 CoC: Project 5
Image-extracted defocus distance, supervised learning

## Project 9 CoC: Project 6
input: video sequence
pass 1: feature extractor - supervised learning
pass 2: action control - supervised learning
pass 3: action control - RL
	
