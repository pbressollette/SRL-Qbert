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
mutable struct Qbert
    ale
    base
    seqs
    function Qbert(k=2)
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
    reset_game(env.ale)
    return Float32.(getRAM(env.ale)) ./ 255.0f0
end

function count_colored_cubes(env::Qbert)
    ram = getRAM(env.ale)
    tile_indices = [22, 53, 55, 84, 86, 88, 99, 101, 103, 105, 
                    2, 4, 6, 8, 10, 33, 35, 37, 39, 41, 43]
    return sum(ram[i] != 148 for i in tile_indices)
end

# Step function
function step!(env::Qbert, action)
    seq = env.seqs[argmax(action)]
    cubes_before = count_colored_cubes(env)
    total_reward = 0.0f0
    done = false
    for idx in seq
        total_reward += act(env.ale, env.base[idx])
        done = game_over(env.ale)
        done && break
    end
    cubes_after = count_colored_cubes(env)
    shaped_reward = total_reward / 25.0f0 + 0.1f0 * Float32(cubes_after - cubes_before)
    return Float32.(getRAM(env.ale)) ./ 255.0f0, shaped_reward, game_over(env.ale)
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
        state = reset!(env)
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
    state = reset!(env)
    done = false
    while !done
        Θ = model(state)
        η = Θ + sigma * Random.randn(Float32, length(Θ))
        action = Qbert_optimization(η)
        state_next, reward, done = step!(env, action)
        push!(
            buffer, (state, Θ, η, action, reward, state_next, done, false)
        )
        state = state_next
    end
    return buffer
end

function generate_episode_uniform(env::Qbert, gamma=0.99f0)
    buffer = []
    state = reset!(env)
    done = false
    while !done
        action = random_solution(env)
        state_next, reward, done = step!(env, action)
        Θ = zeros(Float32, length(env.seqs))
        push!(buffer, (state, Θ, action, action, reward, state_next, done, true))
        state = state_next
    end
    # Compute discounted returns (Monte Carlo targets)
    G = 0.0f0
    for i in length(buffer):-1:1
        t = buffer[i]
        G = t[5] + gamma * G
        buffer[i] = (t[1], t[2], t[3], t[4], G, t[6], t[7], t[8])
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
