# SRL implementation on Qbert

Implementation of a new reinforcement learning paradigm on the Atari game [Qbert](https://fr.wikipedia.org/wiki/Q*bert). 

## Structured Reinforcement Learning

You can click this [link](https://arxiv.org/abs/2505.19053) to understand the details of SRL. 

## Implementation on Qbert

I used the library ALE for Atari games on Julia, the environment is defined in `utils.jl`. 

- **State** — the 128 bytes of the Atari RAM, normalized to `[0, 1]`
- **Actions** — 36 macro-actions, each a sequence of 2 primitive actions from the minimal action set
- **Reward** — game score divided by 25 (one colored cube = +1), plus a shaping bonus of 0.1 per newly colored cube, read from the RAM

## Architecture

The SRL agent follows an actor-critic architecture with a combinatorial optimization (CO) layer embedded in the actor:

- **Actor** — MLP mapping the state `s` to a score vector `θ`
- **CO-layer** — maps `θ` to a one-hot action vector by solving an argmax over the action space
- **Critic** — MLP estimating Q-values from concatenated `(s, a)` pairs, updated via TD learning
- **Fenchel-Young loss** — enables end-to-end backpropagation through the CO-layer

During training, the actor score vector `θ` is perturbed with Gaussian noise, candidate actions are generated via the CO-layer and evaluated by the critic. A softmax-weighted target action `â` is computed and the actor is updated by minimizing the Fenchel-Young loss between `â` and `θ`.

## Training

The training loop follows a standard replay buffer setup with two phases:

1. **Critic warmup** (100 episodes) — episodes are played with a uniform random policy and the critic is trained alone on Monte Carlo returns, to stabilize Q-value estimates before the actor starts updating
2. **Joint training** (50 episodes) — episodes are played by the perturbed actor, the critic is updated with TD targets from a target critic (synced every 5 episodes), and the actor with the Fenchel-Young loss

Training logs policy entropy, actor gradient norm, critic loss and Q-value spread across candidate actions.

## Usage

```julia
include("setup.jl")  # initialize actor/critic and evaluate the random baseline
include("SRL.jl")    # train the SRL agent
include("plots.jl")  # plot training and final results
```

## Results & Limitations

Average shaped reward over 100 evaluation episodes:

| Policy | Mean | Median |
|---|---|---|
| SRL (final) | 8.94 | 8.90 |
| Random | 4.83 | 3.35 |

The final SRL policy scores about 1.85× the random baseline, but the untrained actor already reached a similar level (≈ 9.45 on average during warmup, when the actor is not updated). Most of the gap with the random baseline likely comes from acting deterministically rather than from actor training.

The main bottleneck is the critic: the actor imitates the critic's preferred actions, so when Q-values barely differ across candidates, the actor receives no useful signal. Likely causes and next steps:

- **Too few critic updates** — only 2 gradient steps per episode (300 in total); increasing this is the first thing to try
- **Mixed targets in the replay buffer** — Monte Carlo returns of the random policy coexist with TD targets for the actor's policy
- **Short training** — 150 episodes with a small RAM-based network
- **Weak structure** — on Q\*bert the CO-layer is a plain argmax over 36 actions; SRL is designed for large combinatorial action spaces (routing, scheduling), where it should matter more

## Dependencies

```julia
ArcadeLearningEnvironment  # Atari environment
Flux                       # Neural networks and automatic differentiation
InferOpt                   # Fenchel-Young loss, PerturbedAdditive
JLD2                       # Saving models and results
Plots, StatsPlots          # Figures
```

