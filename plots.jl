include("utils.jl")

using StatsPlots 

# Load data
function load_data()
    log_srl = load(joinpath(logdir, "Qbert_SRL_training_results.jld2"))
    log_bl = load(joinpath(logdir, "Qbert_baselines.jld2"))

    train_data = [log_srl["train_rew"][1:(end - 1)]]

    final_data = [
        log_srl["train_final"][1:(end - 1)], log_bl["random_train"][1:(end - 1)]
    ]

    return train_data, final_data
end

# Basic lineplot
function training_plot(
    data;
    include_legend=true,
    cumulative=true,
    title,
    ylabel_text,
    yticks=nothing,
    size=(800, 400)
)
    # Apply cumulative max if requested
    processed_data = data
    if cumulative
        processed_data = [accumulate(max, d) for d in data]
    end

    plt = plot(
        processed_data[1];
        label="SRL",
        color=:blue,
        linewidth=4,
        xlabel="training episode",
        title=title,
        labelfontsize=16,
        tickfontsize=14,
        size=size
    )

    # Set y-label
    plot!(plt; ylabel=ylabel_text, labelfontsize=16)

    if include_legend
        plot!(plt; legend=:bottomright, legendfontsize=16)
    else
        plot!(plt; legend=:none)
    end

    if !isnothing(yticks)
        yticks!(plt, yticks)
    end

    return plt
end

# Basic boxplot
function create_boxplot(data, colors, labels; title, yticks=nothing, ylims=(-100, 100))
    positions = collect(1:length(data))
    plt = plot(ylims=ylims)
    
    n = length(data)
    for i in 1:n
        boxplot!(
            fill(positions[i], length(data[i])),
            data[i];
            fillcolor=colors[i],
            alpha=0.6,
            linewidth=2,
            linecolor=:black,
            legend=false,
            label="",
            title=title,
            outliers=false,
        )
        scatter!(
            [positions[i]],
            [mean(data[i])];
            markershape=:circle,
            markersize=4,
            markercolor=:black,
            label="",
        )
    end

    # Apply custom yticks if provided
    if !isnothing(yticks)
        yticks!(plt, yticks...)
    end

    xticks!(positions, labels; tickfontsize=14, xrotation=45)
    return plt
end

function boxplot_random(
    data,
    factor; # Factor to account for positive or negative rewards
    figsize=(800, 400),
    log_ticks=[-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3],
    ylimits=(-100, 100),
    ytext="Env: delta random (%)",
)
    ε = eps()
    srl_train, train_benchmark = data[1], data[2]

    # Calculate percentage differences from benchmark (skip entries where benchmark is ~0)
    valid_train = abs.(train_benchmark) .> ε
    train_delta = factor .* (srl_train[valid_train] .- train_benchmark[valid_train]) .* 100 ./ train_benchmark[valid_train]

    # Apply log transform to data
    data_plot = [filter(isfinite, sign.(d) .* log10.(ε .+ abs.(d))) for d in [train_delta]]

    # Calculate original percentage values for tick labels
    orig_ticks = [sign(t) * (10^abs(t) - ε) for t in log_ticks]
    tick_labels = ["$(round(Int, t))" for t in orig_ticks]

    labels = ["SRL"]
    colors = [:red]

    plt = create_boxplot(
        [data_plot[1]],
        colors,
        labels;
        title="train",
        yticks=(log_ticks, tick_labels),
        ylims=ylimits,
    )

    plot!(plt; size=figsize)
    ylabel!(plt, ytext; labelfontsize=16)
    return plt
end

# Data loading
qbert_training, qbert_final = load_data() .* 0.1; # Factor to account for reward scale

# Training plot
qbert_training_plot = training_plot(
    qbert_training;
    include_legend=true,
    cumulative=true,
    title="Q*bert",
    ylabel_text="val. rew. (10^1)",
);
display(qbert_training_plot)
savefig(qbert_training_plot, joinpath(plotdir, "qbert_training_plot.pdf"))

# Results plot
qbert_results_plot = boxplot_random(
    qbert_final,
    1;
    log_ticks=[-2, -1, 0, 1, 2, 3],
    ylimits=(-3, 3.2),
    ytext="Q*bert: delta random (%)",
);
display(qbert_results_plot)
savefig(qbert_results_plot, joinpath(plotdir, "qbert_results_plot.pdf"))