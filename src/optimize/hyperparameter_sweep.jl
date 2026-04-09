"""
Hyperparameter sweep functionality for neural mass model optimization.

This module provides functions to systematically explore hyperparameter spaces for optimization,
including parameter range levels, reparameterization scales, optimizer settings, and more.

Usage: Import ENMEEG and call `run_hyperparameter_sweep()` with a settings file path.
"""

using CSV
using DataFrames
using JSON

function update_settings!(settings::Settings, 
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

    # normalize_fspb_ssvep_weights!(settings.optimization_settings.loss_settings; background_auto=true)
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
                            data::Data,
                            combo::Tuple,
                            combo_keys::Vector{String},
                            combo_idx::Int)

    os = settings.optimization_settings
    update_settings!(settings, combo, combo_keys)
    # Loss settings are now simplified: only region weighting (roi_weight, bg_weight) is used
    println(" ROI weight: $(settings.optimization_settings.loss_settings.roi_weight)")
    println(" Background weight: $(settings.optimization_settings.loss_settings.bg_weight)")
    flush(stdout)
    
    net = build_network(settings)
    set_all_params_tunable!(net.params)

    decision_dim = count_decision_variables(net, os)
    ensure_population_size!(os, decision_dim)
    ensure_sigma0!(os, decision_dim)

    optsol, optlogger, setter, blocks = optimize_network(
        net, data, settings;
        hyperparam_combo=combo, hyperparam_idx=combo_idx, hyperparam_keys=combo_keys
    )
    save_optimization_results(
        optsol, optlogger, setter, net, data, settings;
        blocks=blocks, hyperparam_combo=combo, hyperparam_idx=combo_idx, hyperparam_keys=combo_keys
    )
    return nothing
end

function count_decision_variables(net::Network, os::OptimizationSettings)::Int
    blocks = prepare_optimization_blocks(net, os)
    return length(blocks.tunable_params_symbols) + length(blocks.init_lb)
end

"""
    run_hyperparameter_sweep(settings::Settings, data::Data; combo_idx=nothing, max_runs=nothing)

Run a systematic hyperparameter sweep for neural mass model optimization.

# Arguments
- `settings::Settings`: Settings object with hyperparameter sweep configuration
- `data::Data`: Target PSD data for optimization
- `combo_idx`: Optional specific combination index to run
- `max_runs`: Optional limit on number of hyperparameter combinations to run

"""
function run_hyperparameter_sweep(settings::Union{Nothing, Settings},
                                  data::Data;
                                  combo_idx::Union{Nothing, Int}=nothing,
                                  max_runs::Union{Nothing, Int}=nothing)

    vprint("\n--- RUNNING HYPERPARAMETER SWEEP ---\n", level=1)
    combo_keys, combos = build_sweep_combos(settings)
    total = length(combos)

    if combo_idx !== nothing
        vprint("\n[Hyperparameter Sweep $(combo_idx) / $(length(combos))]", level=1)
        combo = _get_combo_at(combos, combo_idx)
        vprint("Hyperparameter combination: $(combo)\n", level=1)
        _run_single_combo!(settings, data, combo, combo_keys, combo_idx)
        return nothing
    end

    if max_runs !== nothing
        combos = combos[1:min(total, max_runs)]
        vprint("Limiting hyperparameter sweep to $(length(combos)) runs (max_runs=$(max_runs)). Check settings.", level=1)
    end

    for (i, combo) in enumerate(combos)
        vprint("\n[Hyperparameter Sweep $(i) / $(length(combos))]", level=1)
        vprint("Hyperparameter combination: $(combo)\n", level=1)

        combo = _get_combo_at(combos, i)
        _run_single_combo!(settings, data, combo, combo_keys, i)
    end
end
