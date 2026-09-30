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
    episodes,
    iterations,
    batch_size,
    sigmaF_values,
    sigmaB_values,
    lr_values,
    temp_values,
    gamma,
    critic_warmup,
    critic_lr
)

    # Initialize training
    opt_a = Flux.Optimise.Optimiser(Flux.Optimise.ClipNorm(1.0), Flux.Optimise.Adam(lr_values[1]))
    opt_c = Flux.Optimise.Adam(critic_lr)
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / episodes
    lr_step = (lr_values[1] - lr_values[2]) / episodes
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / episodes

    train_rewards = Float64[]
    losses = Float64[] # Can be used to store actor or critic losses
    replay_buffer = []
    rb_position = 1
    rb_size = 0
    no_samples = 40

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, 30)[1])
        @info e, "sigmaF:", sigmaF,
        "lr:", opt_a.os[2].eta,
        "temp:", temp,
        "train:", train_rewards[end]

        ep_actor_norms = Float64[]
        ep_critic_losses = Float64[]
        ep_q_spreads = Float64[]

        # Collect experience
        experience = e <= critic_warmup ?
            generate_episode_uniform(env) :
            generate_episode(env, model, sigmaF)
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
                qvals = [critic(vcat(data[1], a))[1] for a in candidates]
                push!(ep_q_spreads, std(qvals))
                values = qvals ./ temp
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
                gn = sqrt(sum(sum(abs2, grads[p]) for p in Flux.params(model) if grads[p] !== nothing))
                push!(ep_actor_norms, gn)
                Flux.update!(opt_a, Flux.params(model), grads)
            end

            # Critic update
            y = Float32[
                let t = batches[i][j]
                    (t[8] || t[7]) ? t[5] :
                    t[5] + gamma * target_critic(vcat(t[6], Qbert_optimization(model(t[6]))))[1]
                end
                for j in 1:batch_size
            ]
            cl = mean([(critic(vcat(batches[i][j][1], batches[i][j][4]))[1] - y[j])^2 for j in 1:batch_size])
            push!(ep_critic_losses, cl)  
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

        # Compute entropy + log
        action_idx = [argmax(step[4]) for step in experience]
        n = length(action_idx)
        counts = Dict{Int,Int}()
        for ai in action_idx; counts[ai] = get(counts, ai, 0) + 1; end
        H = -sum((c/n) * log(c/n) for c in values(counts))

        @info e, "H:", round(H, digits=3),
                "grad:", isempty(ep_actor_norms) ? 0.0 : round(mean(ep_actor_norms), digits=4),
                "cl:", round(mean(ep_critic_losses), digits=3),
                "qspread:", round(mean(ep_q_spreads), digits=4)

        # Update schedules
        # sigmaF with accelerated decrease post-warmup
        if e <= critic_warmup
            sigmaF = sigmaF_values[1]  # constant during warmup
        elseif e <= critic_warmup + 50
            # rapid decrease over 50 episodes after warmup
            progress = (e - critic_warmup) / 50
            sigmaF = sigmaF_values[1] - progress * (sigmaF_values[1] - sigmaF_values[2])
        else
            sigmaF = sigmaF_values[2]
        end
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
    end

    # Returns
    return model, train_rewards, losses
end

# Train SRL
SRL_model, SRL_train, losses = SRL_Qbert(
    env, SRL_model, critic, target_critic;
    episodes=150,
    iterations=2,
    batch_size=32,
    sigmaF_values=[1.0, 1.0],
    sigmaB_values=[1.0, 0.3],
    lr_values=[1e-5, 1e-6],
    temp_values=[1.5, 1.0],
    gamma=0.99,
    critic_warmup=100,
    critic_lr=1e-3,
)

# Test the trained model
SRL_final_train, SRL_final_train_rew = val_test(env, SRL_model, 100)

# Plot the train and validation rewards
Qbert_SRL_rew_line = plot(SRL_train; label="train history", title="Qbert SRL", marker=:o)
savefig(Qbert_SRL_rew_line, joinpath(plotdir, "Qbert_SRL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "Qbert_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    train_final=SRL_final_train_rew,
)