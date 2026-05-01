using Flux
using Flux.Optimise
using InferOpt
using Random
using Plots
using LinearAlgebra
using ArcadeLearningEnvironment
using JLD2
using Statistics

const logdir = joinpath(@__DIR__, "logs")
mkpath(logdir)
const plotdir = joinpath(@__DIR__, "plots")
mkpath(plotdir)

## Set up the environment

# Environment structure
struct Qbert
    ale
    base
    seqs
    function Qbert(k=3)
        ale = ALE_new()
        loadROM(ale, "qbert")
        base = getMinimalActionSet(ale)
        seqs = vec(collect(Iterators.product(fill(1:length(base), k)...)))
        return new(ale, base, seqs)
    end
end

## Basics operations of environment

# Reset the environment
function reset!(env::Qbert)
    ale = env.ale
    reset_game(ale)
    return getRAM(ale)
end

# Step function 
function step!(env::Qbert, action)
    seq = env.seqs[argmax(action)]
    total_reward = 0.0f0
    done = false
    for idx in seq
        reward = act(env.ale, env.base[idx])
        total_reward += reward
        done = game_over(env.ale)
        done && break
    end
    return Float32.(getRAM(env.ale)), total_reward, game_over(env.ale)
end

# Random solution
function random_solution(env::Qbert)
    a = zeros(Float32, length(env.seqs))
    a[rand(1:length(env.seqs))] = 1.0f0
    return a
end

# Qbert CO-layer
function Qbert_optimization(Θ)
    a = zeros(Float32, length(Θ))
    a[argmax(Θ)] = 1.0f0
    return a
end

## Solution functions

# Random
function random_policy(env::Qbert, episodes)
    rewards = Float64[]
    for _ in 1:episodes
        reset!(env)
        total_reward = 0.0
        done = false
        while !done
            action = random_solution(env)
            _, reward, done = step!(env, action)
            total_reward += reward
        end
        push!(rewards, total_reward)
    end
    return mean(rewards), rewards
end

# Validation/testing function
function val_test(env::Qbert, model, episodes)
    rewards = Float64[]
    for _ in 1:episodes
        state = Float32.(reset!(env))
        total_reward = 0.0
        done = false
        while !done
            action = Qbert_optimization(model(state))
            state, reward, done = step!(env, action)
            total_reward += reward
        end
        push!(rewards, total_reward)
    end
    return mean(rewards), rewards
end

## Model definitions

# Episode generation
function generate_episode(env::Qbert, model, sigma)
    buffer = []
    state = Float32.(reset!(env))
    done = false
    while !done
        Θ = model(state)
        η = Θ + sigma * Random.randn(Float32, length(Θ))
        action = Qbert_optimization(η)
        state_next, reward, done = step!(env, action)
        push!(
            buffer, (state, Θ, η, action, reward, state_next, done)
        )
        state = state_next
    end
    return buffer
end

# Replay buffer
function rb_add(replay_buffer, experience, rb_capacity, rb_position, rb_size)
    for transition in experience
        if rb_size < rb_capacity
            push!(replay_buffer, transition)
        else
            replay_buffer[rb_position] = transition
        end
        rb_position = (rb_position % rb_capacity) + 1
        rb_size = min(rb_size + 1, rb_capacity)
    end
    return replay_buffer, rb_position, rb_size
end

# Sample from the replay buffer
function rb_sample(replay_buffer, batch_size)
    idxs = rand(eachindex(replay_buffer), batch_size)
    return [replay_buffer[i] for i in idxs]
end

# Reward comparison function
function reward_comparison(train_rew, val_rew; minim=1)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in eachindex(train_rew)]
    length(means) >= minim ? means = means[minim:end] : nothing
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == maximum(means)  # Check if it's the largest mean
end






