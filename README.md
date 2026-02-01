# snake-rl

## Project 1 cartpole:

basic cartpole with SB3

## Project 2 cartpole:

image feature extraction: from single image, extract the position and angle
performance:
	
```
97% position error within: -0.0170 to 0.0204
97% angle error (deg) within: -1.3026 to 1.6789
Covariance between position and angle errors: -0.000720
Correlation between position and angle errors: -0.214530
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
finished.
```
	Average Episode Length: 500.00
	Max Episode Length: 500
	Min Episode Length: 500
```

## Project 5 CoC: Project 3
basic CoC with sequence of position state vector, RL

Well trained. Well performs. The `evalute.py` shows optimal policy.
```
------------------------------------------
| train/                  |             |
|    ep_len_mean          | 1.8000      |
|    ep_rew_mean          | 9.3800      |
|    fps                  | 105.0000    |
|    iterations           | 0.0000      |
|    time_elapsed         | 472.4670    |
|    total_timesteps      | 50000.0000  |
|    loss_q               | 23.9323     |
|    loss_pi              | -4.6759     |
|    alpha                | 0.1210      |
------------------------------------------
```


## Project 6 CoC
basic CoC with render()

image source:
`https://www.tommackie.com/blog/10-top-landscape-locations-in-the-usa`

## Project 7 CoC: Project 2
CoC feature extractor, supervised learning

Baseline HRNet:

![alt text][Hrnet]

[Hrnet]: https://github.com/EthanAha-ctrl/snake-rl/blob/main/project7/cls-hrnet.png "HRNet"

Modified HRNet:

![alt text][Arch]

[Arch]: https://github.com/EthanAha-ctrl/snake-rl/blob/main/project7/arch.png "Arch"

```
Epoch 2 [7000/7032] Loss: 0.1298
Epoch 2/2 | Time: 4158.7s
Train Loss: 0.3256 | Acc: 93.68%
Val   Loss: 0.1004 | Acc: 98.23%
Saved Best Model (Acc: 98.23%)
```

![alt text][loss]

[loss]: https://github.com/EthanAha-ctrl/snake-rl/blob/main/project7/Screenshot%20From%202025-12-11%2019-25-25.png "Tensorboard"


## Project 8 CoC: Project 4
input: video sequence

pass 1: feature extractor - supervised learning

pass 2: action control
	
