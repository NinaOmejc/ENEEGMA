"""
Hyperparameter sweep functionality for neural mass model optimization.

This module provides functions to systematically explore hyperparameter spaces for optimization,
including parameter range levels, reparameterization scales, optimizer settings, and more.

Usage: Import ENMEEG and call `run_hyperparameter_sweep()` with a settings file path.
"""

using CSV
using DataFrames
using JSON

function update_settings!(settings::ENMEEG.Settings, 
                          combo::Tuple,
                          combo_keys::Vector{String})

    length(combo) == length(combo_keys) || 
        error("Length of combo ($(length(combo))) does not match length of combo_keys ($(length(combo_keys)))")

    for (i, key) in enumerate(combo_keys)
        val = combo[i]
        parts = split(key, ".")
        obj = settings
        for part in parts[1:end-1]
            hasfield(typeof(obj), Symbol(part)) || error("Hyperparameter key $(key) not found (missing $(part))")
            obj = getfield(obj, Symbol(part))
        end
        last = Symbol(parts[end])
        hasfield(typeof(obj), last) || error("Hyperparameter key $(key) not found (missing $(last))")
        field_T = fieldtype(typeof(obj), last)
        setfield!(obj, last, convert(field_T, val))
    end

    # ENMEEG.normalize_fspb_ssvep_weights!(settings.optimization_settings.loss_settings; background_auto=true)
    return settings
end

function build_sweep_combos(settings::Settings)
    os = settings.optimization_settings
    hs = os.hyperparameter_sweep
    combo_keys = [first(ax.hyperparameter) for ax in hs.hyperparameter_axes]
    combo_values = [collect(ax.values) for ax in hs.hyperparameter_axes]           
    return combo_keys, collect(Iterators.product(combo_values...))     
end

function _get_combo_at(combos, idx::Int)
    total = length(combos)
    (1 <= idx <= total) || error("combo_idx $(idx) out of range (1..$(total)).")
    return combos isa AbstractMatrix ? vec(combos)[idx] : combos[idx]
end

function _run_single_combo!(settings::Settings,
                            data::TargetPSD,
                            combo::Tuple,
                            combo_keys::Vector{String},
                            combo_idx::Int)

    os = settings.optimization_settings
    ENMEEG.update_settings!(settings, combo, combo_keys)
    # ENMEEG.normalize_fspb_ssvep_weights!(settings.optimization_settings.loss_settings; background_auto=true)
    if settings.data_settings.task_type == "ssvep"
        settings.optimization_settings.loss_settings.weight_fspb = settings.optimization_settings.loss_settings.weight_background
        println(" Weight for ssvep: $(settings.optimization_settings.loss_settings.weight_ssvep)")
    end
    println(" Weight for fsbp: $(settings.optimization_settings.loss_settings.weight_fspb)")
    println(" Weight for background: $(settings.optimization_settings.loss_settings.weight_background)")
    flush(stdout)
    
    net = build_network(settings)
    ENMEEG.set_all_params_tunable!(net.params)

    decision_dim = ENMEEG.count_decision_variables(net, os)
    ENMEEG.ensure_population_size!(os, decision_dim)
    ENMEEG.ensure_sigma0!(os, decision_dim)

    optsol, optlogger, setter, blocks = ENMEEG.optimize_network(
        net, data, settings;
        hyperparam_combo=combo, hyperparam_idx=combo_idx, hyperparam_keys=combo_keys
    )
    ENMEEG.save_optimization_results(
        optsol, optlogger, setter, net, data, settings;
        blocks=blocks, hyperparam_combo=combo, hyperparam_idx=combo_idx, hyperparam_keys=combo_keys
    )
    return nothing
end

function count_decision_variables(net::ENMEEG.Network, os::OptimizationSettings)::Int
    blocks = ENMEEG.prepare_optimization_blocks(net, os)
    return length(blocks.tunable_params_symbols) + length(blocks.init_lb)
end

"""
    run_hyperparameter_sweep(settings_path::String; max_runs=nothing, overrides=[], force_model_idx=nothing)

Run a systematic hyperparameter sweep for neural mass model optimization.

# Arguments
- `settings_path`: Path to JSON settings file
- `max_runs`: Optional limit on number of hyperparameter combinations to run
- `overrides`: Vector of KEY=>VALUE pairs to override settings
- `force_model_idx`: Override model index from CSV (if using pre-sampled models)

"""
function run_hyperparameter_sweep(settings::Union{Nothing, Settings},
                                  data::TargetPSD;
                                  combo_idx::Union{Nothing, Int}=nothing,
                                  max_runs::Union{Nothing, Int}=nothing,
                                  force_model_idx::Union{Nothing, Int}=nothing)

    os = settings.optimization_settings
    sweep_settings = os.hyperparameter_sweep
    save_mode = sweep_settings.save_results

    println("\n--- RUNNING HYPERPARAMETER SWEEP ---\n")
    combo_keys, combos = ENMEEG.build_sweep_combos(settings)
    total = length(combos)

    if combo_idx !== nothing
        vprint("\n[Hyperparameter Sweep $(combo_idx) / $(length(combos))]", level=1)
        combo = ENMEEG._get_combo_at(combos, combo_idx)
        vprint("Hyperparameter combination: $(combo)\n", level=1)
        ENMEEG._run_single_combo!(settings, data, combo, combo_keys, combo_idx)
        return nothing
    end

    if max_runs !== nothing
        combos = combos[1:min(total, max_runs)]
        vprint("Limiting hyperparameter sweep to $(length(combos)) runs (max_runs=$(max_runs)). Check settings.")
    end

    for (i, combo) in enumerate(combos)
        vprint("\n[Hyperparameter Sweep $(i) / $(length(combos))]", level=1)
        vprint("Hyperparameter combination: $(combo)\n", level=1)

        combo = ENMEEG._get_combo_at(combos, i)
        ENMEEG._run_single_combo!(settings, data, combo, combo_keys, i)
    end
end
