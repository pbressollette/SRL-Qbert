using Flux
using JLD2
using Random

include("utils.jl")

env = Qbert()

Random.seed!(42)
setInt(env.ale, "random_seed", 42)
#42

# Initial models
initial_model = Chain(
    Dense(128 => 64, relu),
    Dense(64 => 36)
)

critic = Chain(
    Dense(128 + length(env.seqs) => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 1),
    vec
)

jldsave(joinpath(logdir, "Qbert_initial_model.jld2"); actor=initial_model, critic=critic)

# Baseline
random_train, random_train_rew = random_policy(env, 100)

jldsave(joinpath(logdir, "Qbert_baselines.jld2"); random_train=random_train_rew)
