# SRL implementation on Qbert

Implementation of a new reinforcement learning paradigm on the Atari game [Qbert](https://fr.wikipedia.org/wiki/Q*bert). 

## Structured Reinforcement Learning

You can click this [link](https://arxiv.org/abs/2505.19053) to understand the details of SRL. 

## Implementation on Qbert

I used the library ALE for Atari games on Julia, the environment is defined in `utils.jl`. 

## Architecture

The SRL agent follows an actor-critic architecture with a combinatorial optimization (CO) layer embedded in the actor:

- **Actor** — MLP mapping the state `s` to a score vector `θ`
- **CO-layer** — maps `θ` to a one-hot action vector by solving an argmax over the action space
- **Critic** — MLP estimating Q-values from concatenated `(s, a)` pairs, updated via TD learning
- **Fenchel-Young loss** — enables end-to-end backpropagation through the CO-layer

During training, the actor score vector `θ` is perturbed with Gaussian noise, candidate actions are generated via the CO-layer and evaluated by the critic. A softmax-weighted target action `â` is computed and the actor is updated by minimizing the Fenchel-Young loss between `â` and `θ`.

## Training

The training loop follows a standard replay buffer setup with two phases:

1. **Critic warmup** — the critic is trained alone for an initial phase to stabilize Q-value estimates before the actor starts updating
2. **Joint training** — actor and critic are updated together

## Dependencies

```julia
ArcadeLearningEnvironment  # Atari environment
Flux                       # Neural networks
Zygote                     # Automatic differentiation
InferOpt                   # Fenchel-Young loss, PerturbedAdditive
```

