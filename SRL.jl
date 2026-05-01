include("utils.jl")

# Env initialization 
env = Qbert()

initial_model = load(joinpath(logdir, "Qbert_initial_model.jld2"))["actor"]
critic = load(joinpath(logdir, "Qbert_initial_model.jld2"))["critic"]

# SRL setup
target_critic = deepcopy(critic)
regularized_predictor = PerturbedAdditive(
    Qbert_optimization; ε=1.0, nb_samples=20, threaded=true
)
loss = FenchelYoungLoss(regularized_predictor)
SRL_model = deepcopy(initial_model)

# SRL training function
function SRL_Qbert(
    env::Qbert,
    model,
    critic,
    target_critic;
    episodes=200,
    iterations=100,
    batch_size=4,
    sigmaF_values=[2.0, 1.0],
    sigmaB_values=[2.0, 1.0],
    lr_values=[1e-3, 5e-4],
    temp_values=[1.0, 0.5],
    gamma=0.99,
    critic_warmup=80,
)

    # Initialize training
    opt_a = Flux.Optimise.Optimiser(Flux.Optimise.ClipNorm(1.0), Flux.Optimise.Adam(lr_values[1]))
    opt_c = Flux.Optimise.Adam(lr_values[1])
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / episodes
    lr_step = (lr_values[1] - lr_values[2]) / episodes
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / episodes

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = Float64[] # Can be used to store actor or critic losses
    best_model = deepcopy(model)
    best_episode = 0
    replay_buffer = []
    rb_position = 1
    rb_size = 0
    no_samples = 40

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, 30)[1])
        push!(val_rewards, val_test(env, model, 30)[1])
        @info e, "sigmaF:", sigmaF,
        "lr:", opt_a.os[2].eta,
        "temp:", temp,
        "train:", train_rewards[end],
        "val:", val_rewards[end]
        if reward_comparison(train_rewards, val_rewards)
            best_model = deepcopy(model)
            best_episode = e
        end

        # Collect experience
        experience = generate_episode(env, model, sigmaF)
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, experience, 8000, rb_position, rb_size
        )
        batches = [rb_sample(replay_buffer, batch_size) for _ in 1:iterations]

        for i in 1:iterations
            best_actions = []
            for j in 1:batch_size
                data = batches[i][j]
                θ = model(data[1])
                η = [θ + sigmaB * randn(Float32, length(θ)) for _ in 1:no_samples]
                candidates = [Qbert_optimization(η_i) for η_i in η]
                values = [critic(vcat(data[1], a))[1] for a in candidates] ./ temp
                lse = logsumexp(values)
                probs = exp.(values .- lse)
                best_action = sum(probs .* candidates)
                any(isnan.(best_action)) ? best_action = candidates[argmax(values)] : nothing
                push!(best_actions, best_action)
            end

            # Actor update (skipped during critic warmup)
            if e > critic_warmup
                grads = gradient(Flux.params(model)) do
                    l = 0.0f0
                    for j in 1:batch_size
                        l += loss(model(batches[i][j][1]), best_actions[j])
                    end
                    return l
                end
                Flux.update!(opt_a, Flux.params(model), grads)
            end

            # Critic update
            y = Float32[
                batches[i][j][7] ? batches[i][j][5] :
                batches[i][j][5] + gamma * target_critic(vcat(batches[i][j][6], Qbert_optimization(model(batches[i][j][6]))))[1]
                for j in 1:batch_size
            ]

            grads = gradient(Flux.params(critic)) do
                q = [critic(vcat(batches[i][j][1], batches[i][j][4]))[1] for j in 1:batch_size]
                return mean((q .- y) .^ 2)
            end
            Flux.update!(opt_c, Flux.params(critic), grads)
        end

        # Update target critic every 5 episodes
        if e % 5 == 0
            target_critic = deepcopy(critic)
        end

        # Update schedules
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c.eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
    end

    # Final tests
    push!(train_rewards, val_test(env, best_model, 100)[1])
    push!(val_rewards, val_test(env, best_model, 100)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_episode:",
    best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train SRL
SRL_model, SRL_train, SRL_val, losses = SRL_Qbert(
    env,
    SRL_model,
    critic,
    target_critic;
    episodes=1000,
    iterations=2,
    batch_size=32,
    sigmaF_values=[2.0, 1.0],
    sigmaB_values=[1.0, 0.5],
    lr_values=[1e-4, 5e-5],
    temp_values=[1.0, 0.1],
    gamma=0.99,
    critic_warmup=100,
)

# Test the trained model
SRL_final_train, SRL_final_train_rew = val_test(env, SRL_model, 100)
SRL_final_test, SRL_final_test_rew = val_test(env, SRL_model, 100)

# Plot the train and validation rewards
Qbert_SRL_rew_line = plot(SRL_train; label="train history", title="Qbert SRL", marker=:o)
plot!(Qbert_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(Qbert_SRL_rew_line, joinpath(plotdir, "Qbert_SRL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "Qbert_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)