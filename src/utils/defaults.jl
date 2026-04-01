"""
    create_default_settings(; network_name::String="SimpleNetwork", n_nodes::Int=2)

Create a complete Settings object with sensible defaults for all options.

# Arguments
- `network_name::String`: Name for the network (default: "SimpleNetwork")
- `n_nodes::Int`: Number of nodes in the network (default: 2)

# Returns
A fully initialized `Settings` object with:
- General settings: default paths, verbosity
- Network settings: n_nodes with default connectivity
- Simulation settings: time span, solver parameters
- Optimization settings: loss functions, optimizer params

# Example
```julia
settings = create_default_settings(network_name="MyNetwork", n_nodes=3)
settings_dict = settings_to_dict(settings)
save_settings_to_json(settings_dict, "settings.json")
```
"""
function create_default_settings(;
    network_name::String="SimpleNetwork",
    n_nodes::Int=2,
    path_out::String="./results",
    tspan::Tuple{Float64, Float64}=(0.0, 1000.0),
    dt::Float64=0.1
)::Dict{String, Any}
    
    # General settings
    general_settings = Dict{String, Any}(
        "network_name" => network_name,
        "path_out" => path_out,
        "verbose" => false,
        "verbosity_level" => 1,
        "seed" => nothing,
        "make_plots" => false,
        "save_model_formats" => ["json"]
    )
    
    # Network settings - default to all nodes "Unknown" (grammar-sampled)
    node_names = ["N$i" for i in 1:n_nodes]
    node_models = fill("Unknown", n_nodes)
    
    # Default connectivity: weakly connected sequential
    network_conn = zeros(n_nodes, n_nodes)
    for i in 1:(n_nodes-1)
        network_conn[i, i+1] = 0.1
        network_conn[i+1, i] = 0.1
    end
    
    # Default delays: 10ms between connected nodes
    network_delay = fill(0.01, n_nodes, n_nodes)
    
    network_settings = Dict{String, Any}(
        "network_name" => network_name,
        "n_nodes" => n_nodes,
        "node_names" => node_names,
        "node_models" => node_models,
        "node_coords" => [(0.0, float(i)*10.0, 0.0) for i in 1:n_nodes],
        "network_conn" => network_conn,
        "network_delay" => network_delay,
        "network_conn_funcs" => String[],
        "sensory_input_conn" => zeros(Int, n_nodes),
        "sensory_input_func" => "",
        "seed_sensory_input" => nothing,
        "eeg_output" => ""
    )
    
    # Simulation settings
    simulation_settings = Dict{String, Any}(
        "n_runs" => 1,
        "tspan" => collect(tspan),
        "dt" => dt,
        "solver" => "Tsit5",
        "solver_kwargs" => Dict{String, Any}(
            "abstol" => 1e-6,
            "reltol" => 1e-5,
            "maxiters" => 1e5
        ),
        "saveat" => dt
    )
    
    # Data settings (for when running optimization)
    data_settings = Dict{String, Any}(
        "data_path" => "",
        "target_channel" => "IC3",
        "task_type" => nothing
    )
    
    # Loss settings (for optimization)
    loss_settings = Dict{String, Any}(
        "loss_fn" => "psd_iae",
        "freq_bands" => [1.0, 50.0],
        "sigma_meas" => 0.0,
        "noise_seed" => nothing,
        "psd_window_size" => 5,
        "psd_poly_order" => 2,
        "psd_rel_eps" => 1e-12,
        "psd_smooth_sigma" => 1.0
    )
    
    # Optimizer settings
    optimizer_settings = Dict{String, Any}(
        "optim_backend" => "Optimization.jl",
        "solver_type" => "Adam",
        "maxiters" => 100,
        "learning_rate" => 0.01,
        "beta1" => 0.9,
        "beta2" => 0.999
    )
    
    # Optimization settings
    optimization_settings = Dict{String, Any}(
        "loss_settings" => loss_settings,
        "optimizer_settings" => optimizer_settings,
        "reparametrize" => false,
        "abs_target_loss" => 0.1,
        "component_fit" => "all",
        "save_modeled_psd" => false,
        "save_all_optim_restarts_results" => false,
        "hyperparameter_sweep" => Dict{String, Any}()
    )
    
    # Sampling settings (for grammar-based model generation)
    sampling_settings = Dict{String, Any}(
        "grammar_file" => "",
        "sampling_method" => "uniform",
        "random_seed" => nothing
    )
    
    # Assemble complete settings dict
    settings_dict = Dict{String, Any}(
        "general_settings" => general_settings,
        "network_settings" => network_settings,
        "simulation_settings" => simulation_settings,
        "data_settings" => data_settings,
        "optimization_settings" => optimization_settings,
        "sampling_settings" => sampling_settings
    )
    
    return settings_dict
end

"""
    settings_to_dict(settings::Settings)::Dict{String, Any}

Convert a Settings object back to its dictionary representation.
Useful for saving and inspecting configuration.
"""
function settings_to_dict(settings::Settings)::Dict{String, Any}
    d = Dict{String, Any}()
    
    # General settings
    d["general_settings"] = Dict(
        "network_name" => "",
        "path_out" => settings.general_settings.path_out,
        "verbose" => settings.general_settings.verbose,
        "verbosity_level" => settings.general_settings.verbosity_level,
        "seed" => settings.general_settings.seed,
        "make_plots" => settings.general_settings.make_plots
    )
    
    # Network settings
    ns = settings.network_settings
    d["network_settings"] = Dict(
        "network_name" => ns.network_name,
        "n_nodes" => ns.n_nodes,
        "node_names" => ns.node_names,
        "node_models" => ns.node_models,
        "network_conn" => ns.network_conn,
        "network_delay" => ns.network_delay,
        "sensory_input_conn" => ns.sensory_input_conn,
        "sensory_input_func" => ns.sensory_input_func,
        "eeg_output" => ns.eeg_output
    )
    
    # Simulation settings
    ss = settings.simulation_settings
    d["simulation_settings"] = Dict(
        "n_runs" => ss.n_runs,
        "tspan" => collect(ss.tspan),
        "dt" => ss.dt,
        "solver" => ss.solver
    )
    
    return d
end

"""
    save_settings_to_json(settings_dict::Dict, filepath::String)::String

Save settings dictionary to a JSON file.

# Arguments
- `settings_dict::Dict`: Settings dictionary 
- `filepath::String`: Path where to save the JSON file

# Returns
The full path to the saved file
"""
function save_settings_to_json(settings_dict::Dict, filepath::String)::String
    # Ensure directory exists
    dir = dirname(filepath)
    isempty(dir) || mkpath(dir)
    
    # Write JSON
    open(filepath, "w") do f
        JSON.print(f, settings_dict, 2)
    end
    
    vinfo("Settings saved to: $filepath")
    return filepath
end

"""
    load_settings_from_file(filepath::String)::Dict{String, Any}

Load settings from a JSON file with validation.

# Arguments
- `filepath::String`: Path to the JSON settings file

# Returns
Loaded settings dictionary
"""
function load_settings_from_file(filepath::String)::Dict{String, Any}
    !isfile(filepath) && error("Settings file not found: $filepath")
    !endswith(filepath, ".json") && error("Settings file must be .json format")
    
    settings = JSON.parsefile(filepath; dicttype=Dict{String, Any})
    vinfo("Settings loaded from: $filepath")
    return settings
end

"""
    print_settings_summary(settings_dict::Dict; verbosity::Int=1)

Pretty-print a summary of the settings configuration.
"""
function print_settings_summary(settings_dict::Dict; verbosity::Int=1)
    verbosity < 1 && return
    
    gs = get(settings_dict, "general_settings", Dict())
    ns = get(settings_dict, "network_settings", Dict())
    ss = get(settings_dict, "simulation_settings", Dict())
    
    println("\n" * "="^60)
    println("ENEEGMA SETTINGS SUMMARY")
    println("="^60)
    println("\n[General]")
    println("  Network name: $(get(gs, "network_name", "N/A"))")
    println("  Output path:  $(get(gs, "path_out", "N/A"))")
    println("  Verbose:      $(get(gs, "verbose", false))")
    
    println("\n[Network]")
    println("  Nodes:      $(get(ns, "n_nodes", "N/A"))")
    println("  Node names: $(join(get(ns, "node_names", []), ", "))")
    println("  Node models: $(join(get(ns, "node_models", []), ", "))")
    
    println("\n[Simulation]")
    tspan = get(ss, "tspan", nothing)
    if tspan !== nothing
        println("  Time span:  $(tspan[1]) to $(tspan[2]) ms")
    end
    println("  Solver:     $(get(ss, "solver", "Tsit5"))")
    println("  Time step:  $(get(ss, "dt", "N/A")) ms")
    println("  Runs:       $(get(ss, "n_runs", 1))")
    
    println("\n" * "="^60 * "\n")
end
