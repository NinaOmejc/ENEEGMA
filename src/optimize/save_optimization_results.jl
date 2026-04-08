# MAIN FUNCTION
function save_optimization_results(optsol::SciMLBase.OptimizationSolution, 
                                    optlogger::Vector{OptLogEntry},       
                                    setter::Function,  
                                    net::Network, 
                                    target_data::TargetPSD,
                                    settings::Settings;
                                    blocks::Union{Nothing, Any}=nothing,
                                    restart_idx::Union{Int, Nothing}=nothing,
                                    hyperparam_combo::Union{Tuple, Nothing}=nothing,
                                    hyperparam_idx::Union{Int, Nothing}=nothing,
                                    hyperparam_keys::Union{Vector{String}, Nothing}=nothing,
                                    true_parameters::Union{OrderedDict{Num, Float64}, Dict{String, Tuple{Vararg{Float64}}}, OrderedDict{String, Tuple{Vararg{Float64}}}, Nothing}=nothing
                                    )::Nothing

    general_settings = settings.general_settings
    simulation_settings = settings.simulation_settings
    data_settings = settings.data_settings
    optimization_settings = settings.optimization_settings
    loss_settings = optimization_settings.loss_settings

    blocks = blocks === nothing ? prepare_optimization_blocks(net, optimization_settings) : blocks
    params_native, inits_native, min_optlogger_loss = best_params_inits(optsol, optlogger, blocks)
    param_range_entries, init_range_entries = native_range_entries(net, blocks)
    best_loss = min(optsol.minimum, min_optlogger_loss)

    restart_str = restart_idx === nothing ? "" : "_r$(restart_idx)"
    hyperparam_str = hyperparam_idx === nothing ? "" : "_h$(hyperparam_idx)"
    base_prefix = "$(net.name)$(hyperparam_str)$(restart_str)"

    fname_loss = "$(base_prefix)_loss_over_iterations.png"
    path_loss = joinpath(general_settings.path_out, fname_loss)
    plot_loss_over_iterations(optlogger, general_settings, path_loss)
    save_optlogger(optlogger, settings; fullfname_csv=joinpath(general_settings.path_out, "$(base_prefix)_optlogger.csv"))

    # Simulate and evaluate model with best parameters using centralized function
    opt_params = setter(net.problem.p, params_native)
    result = evaluate_model(
        net, opt_params, inits_native, target_data, settings;
        compute_metrics=["loss", "iae", "fsmae", "maef"],
        demean=true
    )
    
    if !result.success
        vprint("Warning: Failed to simulate model with best parameters: $(result.error_msg)")
        vprint("Attempting to save results with fallback values...")
    end
    
    sol = result.sol
    model_prediction = result.model_prediction
    freqs = result.freqs
    modeled_powers = result.model_psd
    
    save_modeled_psd(modeled_powers, freqs, settings; 
                                fullfname_csv=joinpath(general_settings.path_out, "$(base_prefix)_modeled_psd.csv"))

    fname_obs_ts = "$(base_prefix)_plot_obs_vs_mod_ts.png"
    path_obs_ts = joinpath(general_settings.path_out, fname_obs_ts)
    plot_observed_vs_modelled_ts_windows(sol, target_data, general_settings, net;
                                                fullfname_fig=path_obs_ts)

    fname_obs_freq = "$(base_prefix)_plot_obs_vs_mod_freq.png"
    path_obs_freq = joinpath(general_settings.path_out, fname_obs_freq)
    plot_observed_vs_modelled_psd(modeled_powers, target_data.powers, freqs;
                                   fullfname_fig=path_obs_freq,
                                   general_settings=general_settings)

    # PLOT: Parameter exploration
    fname_param = "$(base_prefix)_param_exploration.png"
    path_param = joinpath(general_settings.path_out, fname_param)
    plot_param_exploration(optlogger, net; 
                                true_parameters=true_parameters,
                                loss_settings=loss_settings,
                                fullfname_fig=path_param,
                                general_settings=general_settings)

    # PLOT: Frequency spectra evolution
    fname_freq_plot = "$(base_prefix)_freq_sweep.png"
    fname_freq_anim = "$(base_prefix)_freq_sweep.gif"
    freq_plot_path = joinpath(general_settings.path_out, fname_freq_plot)
    freq_anim_path = joinpath(general_settings.path_out, fname_freq_anim)
    plot_psd_spetra_evolution(optlogger, net, setter;
                    target_data=target_data,
                    general_settings=general_settings,
                    simulation_settings=simulation_settings,
                    loss_settings=loss_settings,
                    fullfname_fig=freq_plot_path,
                    fullfname_anim=freq_anim_path)

    # Use metrics from evaluate_model for consistency with local re-runs
    recomputed_loss = result.success ? get(result.metrics, "loss", best_loss) : best_loss
    r2 = result.success ? get(result.metrics, "r2", NaN) : NaN
    fsmae = result.success ? get(result.metrics, "fsmae", NaN) : NaN
    iae = result.success ? get(result.metrics, "iae", NaN) : NaN
    maef_raw = result.success ? get(result.metrics, "maef", NaN) : NaN
    maef = isnan(maef_raw) ? NaN : round(maef_raw, digits=2)
    
    # Also compute legacy r2 for verification (can be removed later)
    r2_legacy = metric_r2(modeled_powers, target_data.powers)
    if abs(r2 - r2_legacy) > 1e-6
        vprint("Warning: R² mismatch (new=$(r2), legacy=$(r2_legacy))", level=2)
    end
    
    vprint("Best loss (from optimization): $(best_loss)", level=2)
    vprint("Recomputed loss (from eval): $(recomputed_loss)", level=2)
    vprint("FS-MAE: $fsmae", level=2)
    vprint("MAE-F: $maef", level=2)
    vprint("IAE: $iae", level=2)
    best_opt_params_named = OrderedDict{String, Any}()
    if !isempty(blocks.tunable_params_symbols)
        for (sym, val) in zip(blocks.tunable_params_symbols, params_native)
            best_opt_params_named[String(sym)] = val
        end
    end

    duration_sec = (optlogger[end].time - optlogger[1].time).value
    # node_name = isempty(settings.network_settings.node_names) ? "" : join(settings.network_settings.node_names, ",")
    node_models = isempty(settings.network_settings.node_models) ? "" : join(settings.network_settings.node_models, ",")
    task_type = data_settings === nothing ? "" : something(data_settings.task_type, "")

    results = OrderedDict(
        "net_name" => settings.general_settings.exp_name,
        "brain_source" => net.nodes[1].brain_source,
        "exp_name" => general_settings.exp_name,
        "candidate_name" => something(general_settings.candidate_name, "none"),
        "data_file" => something(data_settings.data_file, ""),
        "data_sub" => match(r"sub-\d+", data_settings.data_file) !== nothing ? match(r"sub-\d+", data_settings.data_file).match : "",
        "data_ic" => data_settings.target_channel,
        "restart_idx" => restart_idx,
        "node_models" => node_models,
        "task_type" => task_type,
        "loss" => recomputed_loss,  # Use recomputed loss for consistency
        "loss_best_optim" => best_loss,  # Keep original optimization loss for reference
        "iae" => iae,
        "fsmae" => fsmae,
        "maef" => maef,
        "duration_seconds" => duration_sec,
        "retcode" => optsol.retcode,
        "best_params" => best_opt_params_named,
        "initial_states" => inits_native,
    )

    if hyperparam_combo !== nothing
        results["hyperparam_idx"] = hyperparam_idx
        results["hyperparam_combo"] = hyperparam_combo
        results["hyperparam_keys"] = hyperparam_keys
    end

    if restart_idx === nothing
        results["parameter_ranges"] = param_range_entries
        results["initial_state_ranges"] = init_range_entries
    end

    fname_results = "$(base_prefix)_optimization_results.json"
    results_path = joinpath(general_settings.path_out, fname_results)
    write_compact_json(results_path, results)

end

function best_params_inits(optsol::SciMLBase.OptimizationSolution,
                           optlogger::Vector{OptLogEntry},
                           blocks)
    n_state_vars = length(blocks.initial_values_native)
    n_params = length(optsol.minimizer) - n_state_vars
    n_params < 0 && error("Optimization solution length is inconsistent with state/measurement-noise counts.")
    length(blocks.param_spec) == n_params || error("Mismatch between tunable parameter count and optimization solution length.")
    length(blocks.init_spec) == n_state_vars || error("Mismatch between initial-state count and optimization solution length.")

    decoded_solution = materialize_logged_params(optsol.minimizer, blocks.param_spec, blocks.init_spec)
    params_native = decoded_solution[1:n_params]
    inits_native = decoded_solution[n_params + 1 : n_params + n_state_vars]
    
    return params_native, inits_native, optsol.minimum
end    
#=     
    if isempty(optlogger)
        return params_native, inits_native, optsol.minimum
    end 

    best_optlog_iter = optlogger[argmin([optlogger[i].loss for i in 1:length(optlogger)])]
    if best_optlog_iter.params !== nothing &&
       length(best_optlog_iter.params) == n_params + n_state_vars &&
       round(best_optlog_iter.loss, digits=6) < round(optsol.minimum, digits=6)
        params_native = best_optlog_iter.params[1:n_params]
        inits_native = best_optlog_iter.params[n_params + 1 : n_params + n_state_vars]
        vprint("Using best logged parameters (loss=$(round(best_optlog_iter.loss, digits=6))) instead of optsol.minimizer (loss=$(round(optsol.minimum, digits=6)))",
               level=2)
    end
    return params_native, inits_native, round(best_optlog_iter.loss, digits=6)
end
=#

function save_optlogger(optlogger, settings; fullfname_csv::String="optimization_history.csv")
    if settings.optimization_settings.save_optimization_history
        CSV.write(fullfname_csv, DataFrame(optlogger))
    end
end

function save_modeled_psd(modeled_powers::Vector{Float64},
                          freqs::Vector{Float64},
                          settings::Settings; 
                          fullfname_csv::String="modeled_psd.csv")::Nothing
    if settings.optimization_settings.save_modeled_psd
        CSV.write(fullfname_csv, DataFrame(freq = freqs, psd = modeled_powers))
    end
    return nothing
end

const JSON_SETTINGS_EXCLUDE_FIELDS = Set(["reference_freqs", "reference_psd"])

function write_compact_json(path::AbstractString, data; digits::Int=3)
    prepared = _prepare_json_value(data, digits)
    open(path, "w") do io
        JSON.print(io, prepared, 2)
        println(io)
    end
    vprint("Saved optimization results to JSON: $path")
end

function _prepare_json_value(value, digits::Int)
    return _prepare_json_value(value, digits, IdDict{Any, Bool}())
end

function _prepare_json_value(value, digits::Int, seen::IdDict{Any, Bool})
    if value isa AbstractDict
        container = OrderedDict{Any, Any}()
        for (k, v) in pairs(value)
            if string(k) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[k] = _prepare_json_value(v, digits, seen)
        end
        return container
    elseif value isa NamedTuple
        container = OrderedDict{Any, Any}()
        for (k, v) in pairs(value)
            if string(k) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[String(k)] = _prepare_json_value(v, digits, seen)
        end
        return container
    elseif value isa AbstractVector
        return [_prepare_json_value(value[i], digits, seen) for i in eachindex(value)]
    elseif value isa AbstractMatrix
        rows, cols = size(value)
        out = Vector{Any}(undef, rows)
        for i in 1:rows
            row = Vector{Any}(undef, cols)
            for j in 1:cols
                row[j] = _prepare_json_value(value[i, j], digits, seen)
            end
            out[i] = row
        end
        return out
    elseif value isa AbstractArray
        return _prepare_json_value(collect(value), digits, seen)
    elseif value isa Tuple
        return [_prepare_json_value(val, digits, seen) for val in value]
    elseif value isa Number
        return isfinite(value) ? round(value; digits=digits) : 1e9
    elseif value isa AbstractString || value isa Bool || value === nothing
        return value
    elseif value isa Symbol
        return String(value)
    elseif Base.isstructtype(typeof(value))
        track = Base.ismutabletype(typeof(value))
        if track
            if haskey(seen, value)
                return "[[Circular $(typeof(value))]]"
            end
            seen[value] = true
        end
        container = OrderedDict{Any, Any}()
        for field in fieldnames(typeof(value))
            if String(field) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[String(field)] = _prepare_json_value(getfield(value, field), digits, seen)
        end
        track && delete!(seen, value)
        return container
    else
        return string(value)
    end
end


function plot_observed_vs_modelled_ts(data_predicted::Matrix{Float64},
                                      data_observed::Matrix{Float64}; 
                                      times::Vector{Float64}=Float64[],
                                      xlims::Tuple{<:Real, <:Real}=(0.0, 10.0),
                                      fullfname_fig::String="obs_vs_mod_ts.png",
                                      general_settings::GeneralSettings=nothing
    )::Nothing

    try
        titles = ["Var $i" for i in 1:size(data_observed, 2)]        
        p = plot(layout = (size(data_observed, 2), 2), xlabel="", ylabel="", size=(800, 800))

        # Loop through each subplot and add the observed and predicted data
        for i in 1:size(data_observed, 2)
            if i == 1
                plot!(p[i, 1], times, data_observed[:, i], label="Observed data", ylabel=titles[i], legend=true, xlabel="")
            else
                plot!(p[i, 1], times, data_observed[:, i], label="Observed data", ylabel=titles[i], legend=false, xlabel="")
            end
            plot!(p[i, 1], times, data_predicted[:, i], label="Model prediction", legend=false, xlabel="")
            plot!(p[i, 2], times, data_observed[:, i], label="Observed data", legend=false, xlabel="", xlims=xlims)
            plot!(p[i, 2], times, data_predicted[:, i], label="Model prediction", legend=false, xlabel="", xlims=xlims)
        end
        # add legend to the first subplot

        savefig(p, fullfname_fig);        
    catch
        vprint("\nSimulation failed. Could not plot observed vs modelled time series.\n")
    end

    return nothing
end


"""
plot_observed_vs_modellepsd(sol, data_observed; 
        fullfname_fig, fmin, fmax, 
        use_log_scale, titles)

Plot observed vs. modeled data in frequency domain using periodograms.

# Arguments
- `sol`: The ODE solution
- `data_observed`: Matrix of observed time series data
- `fullfname_fig`: Full file path for saving the figure
- `fmin`: Minimum frequency to display (Hz)
- `fmax`: Maximum frequency to display (Hz)
- `use_log_scale`: Whether to use logarithmic scales (default: true)
- `titles`: Vector of variable titles for subplots

# Returns
- `Nothing`: Saves the plot to the specified path
"""
function plot_observed_vs_modelled_psd(data_modeled::Union{Vector{Float64}, Matrix{Float64}},
                                        data_observed::Union{Vector{Float64}, Matrix{Float64}},
                                        freqs::Vector{Float64};
                                        fullfname_fig::String="obs_vs_mod_freq.png",
                                        general_settings::Union{Nothing, GeneralSettings}=nothing
    )::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    # Check for dimension mismatch (compare frequency axis against first dimension)
    n_freqs = length(freqs)
    n_modeled = data_modeled isa AbstractMatrix ? size(data_modeled, 1) : length(data_modeled)
    n_observed = data_observed isa AbstractMatrix ? size(data_observed, 1) : length(data_observed)

    if n_modeled != n_observed || n_modeled != n_freqs
        vprint("Cannot plot frequency comparison: dimension mismatch (modeled=$n_modeled, observed=$n_observed, freqs=$n_freqs)")
        vprint("  This usually means the simulation was too short or failed. Skipping plot.")
        return nothing
    end
    # Create a figure
    p = plot(layout = (size(data_observed, 2), 1), size=(800, 800),
                legend=:topright, xminorgrid=true, yminorgrid=true)

    plot!(p, freqs, data_modeled, 
            label="Model", 
            xlabel="Frequency (Hz)",
            ylabel="Log Power",
            linewidth=1,
            alpha=1.)

    plot!(p, freqs, data_observed, 
            label="Observed", 
            linewidth=1, 
            alpha=1.)
    
    plot!(p, plot_title="Frequency Domain Comparison", titlefontsize=12)

    savefig(p, fullfname_fig)


    return nothing
end

function plot_observed_vs_modelled_psd(data_modeled::Union{Vector{Float64}, Matrix{Float64}},
                                        target_data::TargetPSD;
                                        fullfname_fig::String="obs_vs_mod_freq.png",
                                        general_settings::Union{Nothing, GeneralSettings}=nothing
    )::Nothing
    return plot_observed_vs_modelled_psd(data_modeled, target_data.powers, target_data.freqs;
                                          fullfname_fig=fullfname_fig,
                                          general_settings=general_settings)
end



function plot_param_exploration(optlogger::Vector{OptLogEntry},
                                net::Network;
                                true_parameters::Union{OrderedDict{Num, Float64}, Dict{String, Tuple{Vararg{Float64}}}, OrderedDict{String, Tuple{Vararg{Float64}}}, Nothing}=nothing,
                                fullfname_fig::String="param_exploration.png",
                                loss_settings::Union{Nothing, LossSettings}=nothing,
                                general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    # Get ALL tunable parameters from network (includes tscale and any other added params)
    tunable_params = get_symbols(get_tunable_params(net.params); sort=true)
    tunable_syms = Symbol.(tunable_params)
    tunable_params_lb, tunable_params_ub = get_param_minmax_values(net.params; p_subset=tunable_params, return_type="vector")

    # if true_parameters is given, use those; otherwise just use midpoints of the bounds
    tunable_params_default_vals =
        true_parameters === nothing ?
            0.5 .* (tunable_params_lb .+ tunable_params_ub) :
            Float64[first(t) for t in values(true_parameters)]

    # Get initial conditions (state variables)
    state_vars = get_state_vars(net.vars)
    inits_lb, inits_ub = get_var_minmax_values(state_vars; return_type="vector")
    inits_default_vals = zeros(length(state_vars.vars))  # assume default inits are zero

    # Combine parameters and initial conditions
    tunables_lb = vcat(tunable_params_lb, inits_lb)
    tunables_ub = vcat(tunable_params_ub, inits_ub)
    tunables_default_vals = vcat(tunable_params_default_vals, inits_default_vals)
    tunable_syms = vcat(tunable_syms, [Symbol(state_vars.vars[i].symbol) for i in 1:length(state_vars.vars)])

    # Collect all sampled tunable params from logger
    has_params = [e.params !== nothing for e in optlogger]
    if !any(has_params)
        vprint("Optlogger does not contain parameter samples; skipping param exploration plot.")
        return nothing
    end

    samples_matrix = reduce(hcat, [e.params for e in optlogger if e.params !== nothing])'
    # rows: samples, cols: parameters
    n_params = size(samples_matrix, 2)
    
    # Verify dimensions match
    expected_n_params = length(tunables_lb)
    if n_params != expected_n_params
        @warn "Parameter dimension mismatch: optlogger has $n_params params, but network has $expected_n_params tunable params + inits. Plot may be incorrect."
        # Use minimum to avoid index errors
        n_params = min(n_params, expected_n_params)
    end

    # one subplot per parameter
    p = plot(layout=(n_params, 1), size=(900, max(250, 200*n_params)),
             legend=:topright, xminorgrid=true, yminorgrid=true)

    unique_restarts = sort(unique(e.irestart for e in optlogger))

    # choose distinct base hues for each restart (avoid black/white)
    base_hues = range(0, stop=360, length=length(unique_restarts)+1)[1:end-1]
    restart_colors = Dict(
        r => HSV(h, 0.8, 0.9) for (r, h) in zip(unique_restarts, base_hues)
    )


    for (j, sym) in enumerate(tunable_syms)
        sp = p[j]

        # scatter (rug-like) of sampled values, colored by restart and iteration
        for r in unique_restarts
            idx = findall(e -> e.irestart == r && e.params !== nothing, optlogger)
            if !isempty(idx)
                xr = [optlogger[i].params[j] for i in idx]
                iter_vals = [optlogger[i].iter for i in idx]

                # normalize iterations to [0, 1] within this restart
                it_min = minimum(iter_vals)
                it_max = maximum(iter_vals)
                it_range = max(it_max - it_min, 1)
                it_norm = [(it - it_min) / it_range for it in iter_vals]

                # alpha: early = 0.2 (very light), late = 1.0 (fully opaque)
                alphas = 0.2 .+ 0.6 .* it_norm
                base_c = restart_colors[r]

                # jitter y so points don't overlap
                yr = 0.0 .+ 0.02 .* (rand(length(xr)) .- 0.5)

                scatter!(sp, xr, yr;
                         markersize=5,
                         markerstrokewidth=0,
                         alpha=alphas,
                         label = j == 1 ? "restart $r" : "")
            end
        end

        # draw min/max (allowed range)
        vline!(sp, [tunables_lb[j]]; color=:black, linestyle=:dash, linewidth=1.5, label = j == 1 ? "min/max" : "")
        vline!(sp, [tunables_ub[j]]; color=:black, linestyle=:dash, linewidth=1.5, label="")

        # draw default value
        vline!(sp, [tunables_default_vals[j]]; color=:red, linestyle=:solid, linewidth=2, label = j == 1 ? "true" : "")

        # cosmetics
        ylims!(sp, (-0.15, 0.15))
        xlabel!(sp, "value")
        ylabel!(sp, "")
        title!(sp, string(sym))
    end

    plot!(p, plot_title = "Parameter exploration across optimization")
    savefig(p, fullfname_fig)

    return nothing
end


"""
    plot_psd_spetra_evolution(optlogger, net, setter; kwargs...)

Simulate and plot the power spectrum associated with each logged parameter
vector (typically from the best restart) so that spectrum evolution across the
optimization can be inspected or animated. Respects the optional σ_meas
measurement-noise parameter by injecting noise prior to PSD computation.
"""
function plot_psd_spetra_evolution(optlogger::Vector{OptLogEntry},
                              net::Network,
                              setter::Function;
                              target_data::Union{TargetPSD, Vector{Float64}, Matrix{Float64}, DataFrame}=Float64[],
                              general_settings::Union{Nothing, GeneralSettings}=nothing,
                              simulation_settings::Union{Nothing, SimulationSettings}=nothing,
                              loss_settings::Union{Nothing, LossSettings}=nothing,
                              fullfname_fig::Union{Nothing, String}=nothing,
                              fullfname_anim::Union{Nothing, String}=nothing,
                              max_entries::Int=typemax(Int))::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    entries = [entry for entry in optlogger if entry.params !== nothing]
    if isempty(entries)
        vprint("Optlogger does not contain parameter samples; skipping frequency-sweep plot.")
        return nothing
    end

    gen = general_settings === nothing ? net.settings.general_settings : general_settings
    sim = simulation_settings === nothing ? net.settings.simulation_settings : simulation_settings
    loss = loss_settings === nothing ? net.settings.optimization_settings.loss_settings : loss_settings

    solver = get_solver(net.problem, sim)
    solver_kwargs = get_solver_kwargs(net.problem, sim)
    n_state_vars = length(net.problem.u0)
    n_params = length(entries[1].params) - n_state_vars

    spectra = NamedTuple{(:iter, :loss, :restart, :freqs, :powers, :sigma)}[]
    for entry in Iterators.take(entries, min(max_entries, length(entries)))
        params_vec = entry.params
        if length(params_vec) != n_params + n_state_vars
            continue
        end

        param_block = params_vec[1:n_params]
        init_block = params_vec[n_params + 1 : n_params + n_state_vars]
        sigma_val = loss.sigma_meas
        new_params = setter(net.problem.p, param_block)
        try
            sol = simulate_problem(net.problem; new_params=new_params, new_inits=init_block,
                                   solver=solver, solver_kwargs=solver_kwargs)
            df = DataFrame(sol)
            if size(df, 2) < 2
                continue
            end
            model_prediction = Matrix(df)[:, 2]
            sampling_rate = length(sol.t) > 1 ? 1.0 / (sol.t[2] - sol.t[1]) : 1.0 / sim.saveat
            freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, sampling_rate, loss)
            # Skip entries with empty or invalid power spectra
            if !isempty(modeled_powers) && all(isfinite, modeled_powers)
                push!(spectra, (iter=entry.iter, loss=entry.loss, restart=entry.irestart,
                                freqs=freqs, powers=modeled_powers, sigma=sigma_val))
            end
        catch err
            vprint("Unable to compute spectrum for iter $(entry.iter): $(err)")
        end
    end

    if isempty(spectra)
        vprint("No spectra computed from optlogger; skipping spectrum sweep plot.")
        return nothing
    end

    # Filter out any spectra with empty powers (safety check)
    spectra = filter(spec -> !isempty(spec.powers) && all(isfinite, spec.powers), spectra)
    if isempty(spectra)
        vprint("All computed spectra had empty or invalid power values; skipping spectrum sweep plot.")
        return nothing
    end

    target_freqs, target_curve = target_data.freqs, target_data.powers
    matches_target = target_curve !== nothing && length(target_curve) == length(spectra[1].freqs)
    overlay_freqs = target_freqs === nothing ? spectra[1].freqs : target_freqs

    power_min = minimum(minimum(spec.powers) for spec in spectra)
    power_max = maximum(maximum(spec.powers) for spec in spectra)
    if matches_target
        power_min = min(power_min, minimum(target_curve))
        power_max = max(power_max, maximum(target_curve))
    end
    span = max(power_max - power_min, eps(Float64))
    lower_pad = 0.05 * span
    upper_pad = 0.1 * span
    ylims_tuple = (power_min - lower_pad, power_max + upper_pad)

    fig_path = isnothing(fullfname_fig) ? "$(gen.path_out)\\$(net.name)_$(gen.exp_name)_freq_sweep.png" : fullfname_fig
    anim_path = isnothing(fullfname_anim) ? "$(gen.path_out)\\$(net.name)_$(gen.exp_name)_freq_sweep.gif" : fullfname_anim

    p = plot(title="Frequency spectra evolution", xlabel="Frequency (Hz)", ylabel="Norm Log10 Power")
    if matches_target
        plot!(p, overlay_freqs, target_curve; label="Target", color=:black, linewidth=2)
    end
    for (idx, spec) in enumerate(spectra)
        label = idx == length(spectra) ? "Iter $(spec.iter)" : ""
        plot!(p, spec.freqs, spec.powers; alpha=idx == length(spectra) ? 0.9 : 0.25, label=label)
    end
    ylims!(p, ylims_tuple)
    savefig(p, fig_path)

    if length(spectra) > 1
        anim = @animate for spec in spectra
            plt = plot(spec.freqs, spec.powers;
                       label="Iter $(spec.iter)",
                       color=:dodgerblue,
                       linewidth=2,
                       xlabel="Frequency (Hz)",
                       ylabel="Norm Log10 Power",
                       title="Restart $(spec.restart) • Iter $(spec.iter) • Loss $(round(spec.loss, digits=4)) • σ=$(round(spec.sigma, digits=4))",
                       ylims=ylims_tuple)
            if matches_target
                plot!(plt, overlay_freqs, target_curve; label="Target", color=:black, linewidth=2, linestyle=:dash)
            end
        end
        try
            gif(anim, anim_path, fps=min(12, max(2, length(spectra) ÷ 10)), verbose=false)
        catch err
            vprint("Unable to save spectrum animation: $(err)")
        end
    end

    return nothing
end


function compute_r2_for_params(net::Network,
                               target_data::TargetPSD,
                               param_symbols::Vector{Symbol},
                               params_native::Vector{Float64},
                               inits_native::Vector{Float64})::Float64
    sim = net.settings.simulation_settings
    loss = net.settings.optimization_settings.loss_settings
    solver = get_solver(net.problem, sim)
    solver_kwargs = get_solver_kwargs(net.problem, sim)

    setter = make_namedtuple_setter(Tuple(param_symbols))
    opt_params = setter(net.problem.p, params_native)
    sol = simulate_problem(net.problem;
                                  new_params=opt_params,
                                  new_inits=inits_native,
                                  solver=solver,
                                  solver_kwargs=solver_kwargs)
    sol.retcode == :Success || return NaN

    df = DataFrame(sol)
    if size(df, 2) < 2
        return NaN
    end
    model_prediction = Matrix(df)[:, 2]
    fs = length(sol.t) > 1 ? 1.0 / (sol.t[2] - sol.t[1]) : 1.0 / sim.saveat
    _, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss)
    target_freqs, target_curve = target_data.freqs, target_data.powers
    if target_curve === nothing || length(target_curve) != length(modeled_powers)
        return NaN
    end
    return metric_r2(modeled_powers, target_curve)
end

function native_range_entries(net::Network, blocks)
    param_ranges = OrderedDict{String, Any}[]
    for (idx, sym) in enumerate(blocks.tunable_params_symbols)
        param_type = String(get_param_by_symbol(net.params, sym).type)
        push!(param_ranges, OrderedDict(
            "name" => String(sym),
            "native_bounds" => [blocks.tunable_params_lb[idx], blocks.tunable_params_ub[idx]],
            "param_type" => param_type
        ))
    end

    init_names = if net.problem.u0 isa NamedTuple
        String.(keys(net.problem.u0))
    else
        ["state_$(i)" for i in 1:length(net.problem.u0)]
    end
    init_ranges = OrderedDict{String, Any}[]
    for i in eachindex(blocks.init_lb)
        push!(init_ranges, OrderedDict(
            "name" => init_names[i],
            "native_bounds" => [blocks.init_lb[i], blocks.init_ub[i]],
        ))
    end
    return param_ranges, init_ranges
end

function plot_loss_over_iterations(optlogger::Vector{OptLogEntry},
                                   general_settings::GeneralSettings,
                                   fullfname_fig::AbstractString)
    general_settings.make_plots || return nothing
    plot([optlogger[i].iter for i in eachindex(optlogger)],
         [optlogger[i].loss for i in eachindex(optlogger)],
         xlabel="Iteration", ylabel="Loss")
    savefig(fullfname_fig)
    return nothing
end

function plot_observed_vs_modelled_ts_windows(sol::SciMLBase.AbstractODESolution,
                                              target_data::Union{TargetPSD, Vector{Float64}, Matrix{Float64}, DataFrame},
                                              general_settings::GeneralSettings,
                                              net::Network;
                                              zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                                              fullfname_fig::Union{Nothing, AbstractString}=nothing)
    general_settings.make_plots || return nothing
    target_data isa TargetPSD || return nothing

    df = DataFrame(sol)
    size(df, 2) >= 2 || return nothing

    model_times = sol.t
    model_ts = Vector(df[:, 2])
    obs_times = target_data.times
    obs_ts = target_data.signal

    p = plot(layout=(2, 1), size=(900, 500), legend=:topright)
    plot!(p[1], obs_times, obs_ts, label="Observed", xlabel="", ylabel="")
    plot!(p[1], model_times, model_ts, label="Model", xlabel="", ylabel="")
    plot!(p[2], obs_times, obs_ts, label="Observed", xlabel="Time (s)", ylabel="", xlims=zoom_window)
    plot!(p[2], model_times, model_ts, label="Model", xlabel="Time (s)", ylabel="", xlims=zoom_window)

    outpath = fullfname_fig === nothing ?
              joinpath(general_settings.path_out,
                       "$(net.name)_$(general_settings.exp_name)_plot_obs_vs_mod_ts.png") :
              String(fullfname_fig)
    savefig(p, outpath)
    return nothing
end


