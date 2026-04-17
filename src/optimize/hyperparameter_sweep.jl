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
        vinfo("  Updated setting: $key = $(getfield(obj, last))"; level=2)
    end

    return settings
end

"""
    show_hyperparameter_combos(settings::Settings; combo_idx=nothing)

Display hyperparameter sweep combinations.

# Arguments
- `settings::Settings`: Settings object with hyperparameter sweep configuration
- `combo_idx::Union{Nothing, Int}`: If provided, show only this specific combination. If nothing, show all.

# Examples
```julia
# Show all combinations
show_hyperparameter_combos(settings)

# Show only combination #3
show_hyperparameter_combos(settings; combo_idx=3)
```
"""
function show_hyperparameter_combos(settings::Settings; combo_idx::Union{Nothing, Int}=nothing)
    combo_keys, combos = build_sweep_combos(settings)
    total = length(combos)
    
    if combo_idx !== nothing
        # Show specific combination
        (1 <= combo_idx <= total) || error("combo_idx $combo_idx out of range (1..$total)")
        println("\n" * "="^80)
        println("HYPERPARAMETER COMBINATION #$combo_idx (of $total)")
        println("="^80)
        combo = _get_combo_at(combos, combo_idx)
        for (key, val) in zip(combo_keys, combo)
            println("  $key = $val")
        end
        println("="^80 * "\n")
    else
        # Show all combinations
        println("\n" * "="^80)
        println("HYPERPARAMETER SWEEP CONFIGURATION")
        println("="^80)
        println("Total combinations to test: $total\n")
        for (i, combo) in enumerate(combos)
            println("Combo $i:")
            for (key, val) in zip(combo_keys, combo)
                println("  $key = $val")
            end
        end
        println("="^80 * "\n")
    end
end

"""
    add_hyperparameter_axis!(settings::Settings, param_path::String, values::Vector)

Add a new hyperparameter sweep axis to the settings.

This is a user-friendly way to add custom hyperparameters to sweep over.

# Arguments
- `settings::Settings`: Settings object
- `param_path::String`: Dot-separated parameter path (e.g., "optimization_settings.optimizer_settings.sigma0")
- `values::Vector`: Vector of values to sweep over

# Example
```julia
# Add a sweep of background loss weight
add_hyperparameter_axis!(settings, "optimization_settings.loss_settings.bg_weight", [0.5, 0.75, 1.0])

# Now settings.optimization_settings.hyperparameter_sweep includes this new axis
```
"""
function add_hyperparameter_axis!(settings::Settings, param_path::String, values::Vector)
    """Add a new hyperparameter to the sweep configuration."""
    settings.optimization_settings.hyperparameter_sweep.hyperparameters[param_path] = Any[v for v in values]
end

"""
Extract hyperparameter sweep combinations from settings.

Each HyperparameterAxis defines one hyperparameter path and its sweep values.
Returns combo_keys (parameter paths) and combos (all value combinations via Cartesian product).
"""
function build_sweep_combos(settings::Settings)
    os = settings.optimization_settings
    hs = os.hyperparameter_sweep
    # Extract parameter paths and their values from hyperparameter dict
    combo_keys = collect(keys(hs.hyperparameters))
    combo_values = [collect(hs.hyperparameters[key]) for key in combo_keys]
    # Create all combinations via Cartesian product
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

    ENEEGMA.update_settings!(settings, combo, combo_keys)
    
    net = build_network(settings)
    set_all_params_tunable!(net.params)

    optimize_network(
        net, data, settings;
        hyperparam_combo=combo, hyperparam_idx=combo_idx, hyperparam_keys=combo_keys
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

    vinfo("RUNNING HYPERPARAMETER SWEEP"; level=1)
    combo_keys, combos = ENEEGMA.build_sweep_combos(settings)
    total = length(combos)
    vinfo("$(length(combos)) hyperparameter combinations to test\n"; level=2)
    
    # Create ONE numbered folder for the entire sweep
    gs = settings.general_settings
    ns = settings.network_settings
    base_output_dir = ENEEGMA.construct_output_dir(gs, ns)
    sweep_output_dir = ENEEGMA.find_next_numbered_folder(base_output_dir, "hyperparam_sweep")
    mkpath(sweep_output_dir)
    settings.optimization_settings.output_dir = sweep_output_dir
    vinfo("Created hyperparameter sweep folder: $sweep_output_dir"; level=1)
    
    # Save settings to the sweep folder
    settings_path = joinpath(sweep_output_dir, "settings.json")
    ENEEGMA.save_settings(settings, settings_path)

    if combo_idx !== nothing
        vinfo("\n[Hyperparameter Sweep $(combo_idx) / $(length(combos))]"; level=1)
        combo = ENEEGMA._get_combo_at(combos, combo_idx)
        vinfo("Hyperparameter combination: $(combo)\n"; level=1)
        ENEEGMA._run_single_combo!(settings, data, combo, combo_keys, combo_idx)
        return nothing
    end

    if max_runs !== nothing
        combos = combos[1:min(total, max_runs)]
        vinfo("Hyperparameter sweep limited to $(length(combos)) runs (max_runs=$(max_runs))"; level=1)
    end

    for (i, combo) in enumerate(combos)
        vinfo("[Hyperparameter sweep $i/$(length(combos))]: $(combo)"; level=1)

        combo = ENEEGMA._get_combo_at(combos, i)
        ENEEGMA._run_single_combo!(settings, data, combo, combo_keys, i)
    end
end
