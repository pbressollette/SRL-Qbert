using Flux
using JLD2
using Random

include("utils.jl")

Random.seed!(5)

env = Qbert()

# Initial models
initial_model = Chain(
    Dense(128 => 64, relu),
    Dense(64 => 32, relu),
    Dense(32 => length(env.seqs))
)

critic = Chain(
    Dense(128 + length(env.seqs) => 64, relu),
    Dense(64 => 32, relu),
    Dense(32 => 1),
    vec
)

jldsave(joinpath(logdir, "Qbert_initial_model.jld2"); actor=initial_model, critic=critic)

# Baselines
random_train, random_train_rew = random_policy(env, 100)
random_test, random_test_rew = random_policy(env, 100)

jldsave(joinpath(logdir, "Qbert_baselines.jld2"); random_train=random_train_rew, random_test=random_test_rew)
