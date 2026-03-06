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
	
## Project a CoC: Project 5, Project 8

input: offline dataset of tensor 10x20x15x10 from 1.1K images.
10 position of CoC
20x15 spatial resolution
10 one-hot encoding of the class of CoC

goal : train the MLP layers to guess the number

## Project b Sharpness: Project a

input: offline dataset of the sharpness, 10x20x15 from 1.1K images
10 position of CoC
20x15 spatial resolution

goal: train the MLP layers to guess the number

## Project c CoC + Sharpness: Project a, Project b

input: offline dataset of the sharpness, 10x20x15, and tensor 10x20x15x10 from 1.1K images
10 position of CoC
20x15 spatial resolution of sharpness, spatial resolution of the tensor
10 one-hot encoding of the class of CoC

## Project d, CoC + Sharpness + SpatialTemporal Transformer: with foreground and background blur

has bug:
https://github.com/EthanAha-ctrl/snake-rl/issues/2

2.5D scene changed from single plane to 2 planes: foreground and background.

the output from hrnet: an 15x20x10 tensor is concatenated with 15x20x1 sharpness value, into a 15x20x11 tensor.

This tensor is stacked over the 10 history frames: giving 15x20x10 tokens, with 11 dim token.

This thing is fed into the 2 layer transformer.

A linear projector makes the 11 dim token into 32 dim.

Passing through a Qwen3-vl alike spatial temporal position encoding, followed by 4x FFN, single head attention, 2 layers.

Then followed by a fully-connection layer to 256 dim vector.

Then passing through the freezed MLP layers.

The training is a supervised behavior cloning: loss = 100 * mse(new_guess, old_guess) + logistic(new_trigger, old_trigger)

The training takes exessively long time: 200K-ish steps in 16 hours on a single 5090 card.

The resulting performance is somewhat reasonable: 4 failures out of 10 episode.

```
New best internal score (minimized loss): -0.0022. Saved to sac_coc_best.pth
------------------------------------------
| train/                  |             |
|    ep_len_mean          | 0.0000      |
|    ep_rew_mean          | 0.0000      |
|    fps                  | 4.0000      |
|    iterations           | 0.0000      |
|    time_elapsed         | 11403.1586  |
|    total_timesteps      | 48250.0000  |
|    loss_guess           | 0.0022      |
|    loss_trigger         | 0.0068      |
|    loss_bc              | 0.0022      |
|    score                | -0.0022     |
------------------------------------------
```

## Project e, CoC + Sharpness + SpatialTemporal Transformer: MLP fine tune: Project d

Based on project d, freeze transformer and unfreeze MLP. Let MLP learns the a) noise of transformer; b) can't fully trust and stop by looking at the sharpness only.

## Project f, replacing ideal circular CoC with captured CoC

(TBD)

## Project g, quantization to 8 bit and port to ONNX runtime

(TBD)

## Project h, video mode, emphasis on smoothness and motion tracking

(TBD)
