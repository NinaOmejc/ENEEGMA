function _coerce_settings_idx_value(val)::Union{Nothing, Int}
    val === nothing && return nothing
    if val isa Integer
        return Int(val)
    elseif val isa AbstractString
        parsed = tryparse(Int, strip(val))
        return parsed === nothing ? nothing : Int(parsed)
    elseif val isa Real
        return Int(round(val))
    end
    return nothing
end

function _dict_to_matrix(raw::AbstractDict, ::Type{T}) where T
    size_raw = get(raw, "size", nothing)
    ref_raw = get(raw, "ref", nothing)
    mem_raw = ref_raw isa AbstractDict ? get(ref_raw, "mem", nothing) : get(raw, "mem", nothing)
    if size_raw === nothing || mem_raw === nothing
        return nothing
    end
    dims = Int.(round.(Float64.(size_raw)))
    length(dims) == 2 || return nothing
    flat = mem_raw isa AbstractVector ? mem_raw : [mem_raw]
    data = if T == String
        String.(flat)
    else
        T.(flat)
    end
    return reshape(data, dims...)
end

# Default grammar file location
const DEFAULT_GRAMMAR = joinpath(pkgdir(@__MODULE__), "grammars", "default_grammar.cfg")

"""    GeneralSettings

General experiment and output configuration.

# Fields
- `exp_name::String`: Experiment/project name used for output file naming.
- `path_out::String`: Base directory where all outputs are saved.
- `save_model_formats::Vector{String}`: Output formats for network equations (e.g., "tex", "pdf").
- `make_plots::Bool`: Whether to generate visualization plots.
- `verbosity_level::Int64`: Logging verbosity (0=silent, 1=minimal, 2=detailed).
- `seed::Union{Int, Nothing}`: Master random seed for reproducibility.
"""
mutable struct GeneralSettings <: AbstractSettings
    exp_name::String
    path_out::String
    save_model_formats::Vector{String}
    make_plots::Bool
    verbosity_level::Int64
    seed::Union{Int, Nothing}

    function GeneralSettings(dict::Dict{String, Any})::GeneralSettings

        # Use safe lookups for possibly-missing sections
        gendict = get(dict, "general_settings", Dict{String, Any}())
        
        # Get experiment name with explicit default
        exp_name = String(get(gendict, "exp_name", "example-exp"))

        # Handle path out: <path_out>/
        # All outputs go to a single base directory
        path_out = String(get(gendict, "path_out", joinpath(".", "results")))

        if !isdir(path_out)
            mkpath(path_out)
        end

        # Handle seed 
        seed = get(gendict, "seed", nothing)
        if seed === nothing || seed == ""
            seed = nothing
        else
            seed = Int(seed)
            Random.seed!(seed)
        end

        function _coerce_level(val, fallback)
            val === nothing && return fallback
            if val isa Integer
                return Int(clamp(val, 0, 2))
            elseif val isa Real
                return Int(clamp(round(Int, val), 0, 2))
            elseif val isa AbstractString
                parsed = tryparse(Int, strip(val))
                return parsed === nothing ? fallback : Int(clamp(parsed, 0, 2))
            else
                return fallback
            end
        end

        # Get verbosity level (0=silent, 1=minimal, 2=detailed)
        raw_level = get(gendict, "verbosity_level", 1)
        verbosity_level = _coerce_level(raw_level, 1)
        gendict["verbosity_level"] = verbosity_level

        new(exp_name,
            path_out,
            Vector{String}(get(gendict, "save_model_formats", ["tex"])),
            Bool(get(gendict, "make_plots", true)),
            verbosity_level,
            seed
        )
    end
end

"""    NetworkSettings

Neural network topology and dynamics configuration.

# Fields
- `name::String`: Network name.
- `n_nodes::Int64`: Number of nodes/populations in the network.
- `node_names::Vector{String}`: Names for each node.
- `node_models::Vector{Union{String, RuleTree}}`: Model type for each node (e.g., "WC", "FHN").
- `node_coords::Vector{Tuple{Float64, Float64, Float64}}`: 3D coordinates for each node.
- `network_conn::Matrix{Float64}`: Connection strength matrix (n_nodes × n_nodes).
- `network_conn_funcs::Matrix{String}`: Connection function strings (n_nodes × n_nodes).
- `network_delay::Matrix{Float64}`: Synaptic delay matrix in ms (n_nodes × n_nodes).
- `sensory_input_conn::Vector{Int64}`: Binary vector indicating which nodes receive sensory input.
- `sensory_input_func::String`: Function string for sensory input generation.
- `sensory_seed::Union{Int, Nothing}`: Random seed for sensory input.
- `init_seed::Union{Int, Nothing}`: Random seed for initial conditions.
- `eeg_output::String`: EEG measurement function expression.
"""
mutable struct NetworkSettings <: AbstractSettings
    name::String
    n_nodes::Int64
    node_names::Vector{String}
    node_models::Vector{Union{String, RuleTree}}
    node_coords::Vector{Tuple{Float64, Float64, Float64}}
    network_conn::Matrix{Float64}
    network_conn_funcs::Matrix{String}
    network_delay::Matrix{Float64}
    sensory_input_conn::Vector{Int64}
    sensory_input_func::String
    sensory_seed::Union{Int, Nothing}
    init_seed::Union{Int, Nothing}
    eeg_output::String

    # Constructor with validation built-in
    function NetworkSettings(settings::Dict{String, Any})::NetworkSettings
        # Get network settings dict
        netsett = get(settings, "network_settings", Dict{String, Any}())
        
        # Get network name with explicit default and better error message
        name_raw = get(netsett, "name", "example-net")
        name = try
            String(name_raw)
        catch e
            throw(ArgumentError("NetworkSettings: 'name' must be a string, got $(typeof(name_raw)). Value: $name_raw"))
        end
        
        # n_nodes is required or defaults to 1
        n_nodes = Int64(get(netsett, "n_nodes", 1))
        
        # Get or create node names - must match n_nodes
        node_names_raw = get(netsett, "node_names", nothing)
        node_names = if isnothing(node_names_raw) || isempty(node_names_raw)
            ["N$i" for i in 1:n_nodes]
        else
            try
                nn = Vector{String}(node_names_raw)
                nn
            catch e
                throw(ArgumentError("NetworkSettings: 'node_names' must be an array of strings, got $(typeof(node_names_raw)). " *
                    "Value: $node_names_raw. Each element must be a string (e.g., [\"N1\", \"N2\"])."))
            end
        end
        
        # Get or create node models - must match n_nodes (supports both String and RuleTree)
        node_models_raw = get(netsett, "node_models", nothing)
        node_models = if isnothing(node_models_raw) || isempty(node_models_raw)
            fill("MPR", n_nodes)
        else
            # Accept both strings and RuleTree objects
            # Check if all elements are strings using explicit iteration for robustness
            is_all_strings = true
            for x in node_models_raw
                if !isa(x, String)
                    is_all_strings = false
                    break
                end
            end
            nm = if is_all_strings
                try
                    Vector{String}(node_models_raw)
                catch e
                    throw(ArgumentError("NetworkSettings: 'node_models' must be an array of strings, got $(typeof(node_models_raw)). " *
                        "Value: $node_models_raw. Each element must be a string (e.g., [\"WC\", \"HH\"])."))
                end
            else
                # Mixed or all RuleTree: keep as Vector{Union{String, RuleTree}}
                Vector{Union{String, RuleTree}}(node_models_raw)
            end
            nm
        end
        
        # Get or create node coordinates - must match n_nodes
        node_coords_raw = get(netsett, "node_coords", nothing)
        node_coords = if isnothing(node_coords_raw) || isempty(node_coords_raw)
            [(0.0, float(i)*10.0, 0.0) for i in 1:n_nodes]
        else
            nc = [(Float64(c[1]), Float64(c[2]), Float64(c[3])) for c in node_coords_raw]
            nc
        end
        
        # Get or create connectivity matrix - must be n_nodes × n_nodes
        network_conn_raw = get(netsett, "network_conn", nothing)
        network_conn = if isnothing(network_conn_raw) || isempty(network_conn_raw)
            zeros(n_nodes, n_nodes)
        else
            nc_mat = if network_conn_raw isa AbstractDict
                parsed = _dict_to_matrix(network_conn_raw, Float64)
                parsed === nothing ? zeros(n_nodes, n_nodes) : parsed
            else
                Matrix{Float64}(hcat(network_conn_raw...)')
            end
            nc_mat
        end
        
        # Get or create connection functions matrix - must be n_nodes × n_nodes
        network_conn_funcs_raw = get(netsett, "network_conn_funcs", nothing)
        network_conn_funcs = if isnothing(network_conn_funcs_raw) || isempty(network_conn_funcs_raw)
            # Default: "linear" for off-diagonal (inter-node), "" for diagonal (no self-connection)
            ncf = fill("", n_nodes, n_nodes)
            for i in 1:n_nodes, j in 1:n_nodes
                if i != j
                    ncf[i, j] = "linear"
                end
            end
            ncf
        else
            ncf = if network_conn_funcs_raw isa AbstractDict
                parsed = _dict_to_matrix(network_conn_funcs_raw, String)
                if parsed === nothing
                    # Default: "linear" for off-diagonal, "" for diagonal
                    ncf = fill("", n_nodes, n_nodes)
                    for i in 1:n_nodes, j in 1:n_nodes
                        if i != j
                            ncf[i, j] = "linear"
                        end
                    end
                    ncf
                else
                    parsed
                end
            else
                permutedims(Matrix{String}(hcat(network_conn_funcs_raw...)))
            end
            ncf
        end
        
        # Get or create delay matrix - must be n_nodes × n_nodes
        network_delay_raw = get(netsett, "network_delay", nothing)
        network_delay = if isnothing(network_delay_raw) || isempty(network_delay_raw)
            zeros(n_nodes, n_nodes)
        else
            nd_mat = if network_delay_raw isa AbstractDict
                parsed = _dict_to_matrix(network_delay_raw, Float64)
                parsed === nothing ? zeros(n_nodes, n_nodes) : parsed
            else
                Matrix{Float64}(hcat(network_delay_raw...)')
            end
            nd_mat
        end
        
        # Get or create sensory input connectivity - must match n_nodes
        sensory_input_conn = let v = get(netsett, "sensory_input_conn", nothing)
            if isnothing(v) || isempty(v)
                ones(Int, n_nodes)  # Default: all nodes receive input
            else
                try
                    sic = Int.(v)
                    sic
                catch e
                    throw(ArgumentError("NetworkSettings: 'sensory_input_conn' must be an array of integers, got $(typeof(v)). " *
                        "Value: $v. Each element must be 0 or 1 (e.g., [1, 0] means N1 gets SI, N2 doesn't)."))
                end
            end
        end

        # Get sensory input function
        sensory_input_func_raw = get(netsett, "sensory_input_func", "rand(Normal(0.0, 1.0))")
        sensory_input_func = try
            String(sensory_input_func_raw)
        catch e
            throw(ArgumentError("NetworkSettings: 'sensory_input_func' must be a string, got $(typeof(sensory_input_func_raw)). " *
                "Value: $sensory_input_func_raw. " *
                "This should be a string expression like 'sin(t)' or 'rand()'."))
        end

        # Get seed for sensory input
        sensory_seed = get(netsett, "sensory_seed", nothing)
        if sensory_seed === nothing || sensory_seed == ""
            sensory_seed = nothing
        else
            sensory_seed = Int(sensory_seed)
        end

        # Get seed for init sampling
        init_seed = get(netsett, "init_seed", nothing)
        if init_seed === nothing || init_seed == ""
            init_seed = nothing
        else
            init_seed = Int(init_seed)
        end

        # Get EEG output function
        eeg_output_raw = get(netsett, "eeg_output", "")
        eeg_output = try
            String(eeg_output_raw)
        catch e
            throw(ArgumentError("NetworkSettings: 'eeg_output' must be a string, got $(typeof(eeg_output_raw)). " *
                "Value: $eeg_output_raw. " *
                "This should be a string expression for EEG output calculation."))
        end

        return new(
            name,
            n_nodes,
            node_names,
            node_models,
            node_coords,
            network_conn,
            network_conn_funcs,
            network_delay,
            sensory_input_conn,
            sensory_input_func,
            sensory_seed,
            init_seed,
            eeg_output
        )
    end
end

"""    SamplingSettings

Grammar-based network topology sampling configuration.

# Fields
- `grammar_file::String`: Path to grammar file (.cfg format).
- `n_samples::Int`: Number of network topologies to sample.
- `only_unique::Bool`: Whether to filter out duplicate samples.
- `grammar_seed::Union{Int, Nothing}`: Random seed for grammar rule selection.
"""
mutable struct SamplingSettings <: AbstractSettings
    grammar_file::String
    n_samples::Int        
    only_unique::Bool
    grammar_seed::Union{Int, Nothing}

    function SamplingSettings(dict::Dict{String, Any})::SamplingSettings
        sampdict = get(dict, "sampling_settings", Dict{String, Any}())
        
        # Construct grammar_file from path + filename (or use combined path if provided)
        grammar_file_raw = get(sampdict, "grammar_file", DEFAULT_GRAMMAR)
        grammar_file = grammar_file_raw === nothing || grammar_file_raw == "" ? DEFAULT_GRAMMAR : String(grammar_file_raw)

        # Parse grammar seed - allow null/empty to mean "random"
        grammar_seed_raw = get(sampdict, "grammar_seed", nothing)
        grammar_seed = if grammar_seed_raw === nothing || grammar_seed_raw == ""
            nothing
        else
            Int(grammar_seed_raw)
        end
        
        new(
            grammar_file,
            Int(get(   sampdict, "n_samples", 10)),
            Bool(get(  sampdict, "only_unique", true)),
            grammar_seed
        )
    end
end


"""    SimulationSettings

ODE solver and time-stepping configuration.

# Fields
- `tspan::Tuple{Float64, Float64}`: Simulation time span [start, end] in milliseconds.
- `dt::Union{Float64, Nothing}`: Fixed time step (ms), required for stochastic solvers.
- `saveat::Float64`: Output sampling rate (ms).
- `abstol::Union{Float64, Nothing}`: Absolute tolerance for adaptive solvers.
- `reltol::Union{Float64, Nothing}`: Relative tolerance for adaptive solvers.
- `solver::Union{String, Nothing}`: ODE solver algorithm (e.g., "Tsit5", "Rosenbrock23").
- `maxiters::Union{Int, Nothing}`: Maximum iterations per step.
"""
mutable struct SimulationSettings <: AbstractSettings
    tspan::Tuple{Float64, Float64}
    dt::Union{Float64, Nothing}
    saveat::Float64
    abstol::Union{Float64, Nothing}
    reltol::Union{Float64, Nothing}
    solver::Union{String, Nothing}
    maxiters::Union{Int, Nothing}
    
    # Constructor with type conversion and default values
    function SimulationSettings(dict::Dict{String, Any})::SimulationSettings

        simdict = get(dict, "simulation_settings", Dict{String, Any}())

        new(
            Tuple{Float64, Float64}(get(simdict, "tspan", (0.0, 59.99609375))),  # Default: 15360 points at 256 Hz (1/256 Hz sampling)
            Float64(get(simdict, "dt", 0.0001)),
            Float64(get(simdict, "saveat", 0.00390625)), # default 1/256 sec for good PSD resolution up to ~100 Hz
            get(simdict, "abstol", nothing),
            get(simdict, "reltol", nothing),
            String(get(simdict, "solver", "Tsit5")),
            get(simdict, "maxiters", nothing)
        )
    end
end


"""
    OptimizerSettings

Configuration specific to optimization algorithms (CMAES).

# Fields
- `population_size`: Population size (λ parameter) for CMAES
- `sigma0`: Initial step-size for CMAES (controls initial search radius)

"""
mutable struct OptimizerSettings <: AbstractSettings
    population_size::Int64
    sigma0::Float64

    function OptimizerSettings(dict::Dict)
        new(
            Int64(get(dict, "population_size", -1)),
            Float64(get(dict, "sigma0", -1.0))
        )
    end
end

# ------------------------------------------------------------------
# PSD preprocessing helpers shared across loss settings and utilities
# ------------------------------------------------------------------

# PSD preprocessing: minimal token parsing for log support
const _PSD_PREPROC_SPLIT_REGEX = r"[\,\s\-\|>]+"
const _LOSSSET_PSD_LOG_TOKENS = Set(["log", "log10", "log2", "ln"])

function _psd_preproc_tokens(spec::Union{Nothing, AbstractString})
    spec === nothing && return String[]
    cleaned = lowercase(strip(String(spec)))
    cleaned == "" && return String[]
    tokens = split(cleaned, _PSD_PREPROC_SPLIT_REGEX; keepempty=false)
    filtered = String[]
    for token in tokens
        t = strip(replace(token, "--" => "-"))
        isempty(t) && continue
        (t == "none" || t == "raw") && return String[]
        push!(filtered, t)
    end
    return filtered
end

function _canonicalize_psd_preproc_string(spec::Union{Nothing, AbstractString})
    tokens = _psd_preproc_tokens(spec)
    isempty(tokens) && return "none"
    return join(tokens, "-")
end

function _psd_preproc_tokens_flags(tokens::Vector{String})
    has_log = any(token -> token in _LOSSSET_PSD_LOG_TOKENS, tokens)
    return has_log
end

function _losssettings_ensure_dict(val)::Union{Dict{String, Any}, Nothing}
    val isa AbstractDict || return nothing
    return Dict{String, Any}(val)
end



function _losssettings_as_bool(val, default::Bool=true)
    val === nothing && return default
    if val isa Bool
        return val
    elseif val isa Integer
        return val != 0
    elseif val isa Real
        return val != 0
    elseif val isa AbstractString
        lowered = lowercase(strip(val))
        lowered in ("1", "true", "yes", "on") && return true
        lowered in ("0", "false", "no", "off") && return false
    end
    return default
end



function normalize_loss_settings_dict!(lossdict::Dict{String, Any})
    # Minimal parsing: extract Welch/noise parameters from nested psd/time_noise sections if present
    psd = _losssettings_ensure_dict(get(lossdict, "psd", nothing))
    if psd !== nothing
        if haskey(psd, "fmin")
            lossdict["fmin"] = Float64(psd["fmin"])
        end
        if haskey(psd, "fmax")
            lossdict["fmax"] = Float64(psd["fmax"])
        end
        welch = _losssettings_ensure_dict(get(psd, "welch", nothing))
        if welch !== nothing
            haskey(welch, "window_sec") && (lossdict["psd_welch_window_sec"] = Float64(welch["window_sec"]))
            haskey(welch, "overlap") && (lossdict["psd_welch_overlap"] = Float64(welch["overlap"]))
            haskey(welch, "nperseg") && (lossdict["psd_welch_nperseg"] = Int(welch["nperseg"]))
            haskey(welch, "nfft") && (lossdict["psd_welch_nfft"] = Int(welch["nfft"]))
        end
        haskey(psd, "noise_avg_reps") && (lossdict["psd_noise_avg_reps"] = Int(psd["noise_avg_reps"]))
    end

    # Remove old noise fields if they exist (moved to DataSettings)
    delete!(lossdict, "psd_noise_avg_reps")
    delete!(lossdict, "measurement_noise_std")
    delete!(lossdict, "loss_noise_seed")

    return lossdict
end

"""    LossSettings

Loss function configuration for optimization.

# Fields
- `fmin::Float64`: Minimum frequency (Hz) for PSD analysis.
- `fmax::Float64`: Maximum frequency (Hz) for PSD analysis.
- `roi_weight::Float64`: Weight for region of interest in loss computation.
- `bg_weight::Float64`: Weight for background activity in loss computation.
- `loss_abstol::Float64`: Absolute tolerance for loss convergence.
- `loss_reltol::Float64`: Relative tolerance for loss convergence.
- `abs_target_loss::Float64`: Absolute loss target for early stopping.
"""
mutable struct LossSettings <: AbstractSettings
    # Frequency analysis range (for loss computation only; PSD settings now in DataSettings.psd)
    fmin::Float64                          # minimum frequency (Hz)
    fmax::Float64                          # maximum frequency (Hz)
    
    # Region weighting (for roi vs background loss)
    roi_weight::Float64                    # weight for region of interest
    bg_weight::Float64                     # weight for background
    
    # Loss convergence criteria (moved from OptimizationSettings)
    loss_abstol::Float64                   # Absolute tolerance for loss
    loss_reltol::Float64                   # Relative tolerance for loss
    abs_target_loss::Float64               # Absolute loss target for early stopping

    function LossSettings(dict::Dict{String, Any})::LossSettings
        cooked = Dict{String, Any}(dict)
        normalize_loss_settings_dict!(cooked)
        dict = cooked
        
        fmin = Float64(get(dict, "fmin", 1.0))
        fmax = Float64(get(dict, "fmax", 45.0))
        
        roi_weight = max(Float64(get(dict, "roi_weight", 1.0)), 0.0)
        bg_weight = max(Float64(get(dict, "bg_weight", 1.0)), 0.0)
        
        loss_abstol = Float64(get(dict, "loss_abstol", 1e-3))
        loss_reltol = Float64(get(dict, "loss_reltol", 1e-3))
        abs_target_loss = max(Float64(get(dict, "abs_target_loss", 0.01)), 0.0)

        new(fmin, fmax, roi_weight, bg_weight, loss_abstol, loss_reltol, abs_target_loss)
    end
end

const _REPARAM_TRUE_STRINGS = Set(["true", "on", "yes", "1"])
const _REPARAM_FALSE_STRINGS = Set(["false", "off", "no", "0"])

function _strategy_from_value(val)
    val === nothing && return nothing
    sym = Symbol(lowercase(strip(String(val))))
    sym in (:typed, :perparam, :per_param, :perparameter, :auto, :modern) && return :typed
    sym in (:none, :off, :disabled) && return :none
    return nothing
end

function _parse_reparam_inputs(raw_flag, raw_strategy)
    strategy_hint = _strategy_from_value(raw_strategy)
    if raw_flag isa Bool
        if !raw_flag || strategy_hint === :none
            return false, :none
        end
        return true, strategy_hint === nothing ? :typed : strategy_hint
    elseif raw_flag isa AbstractString
        lowered = lowercase(strip(raw_flag))
        as_strategy = _strategy_from_value(lowered)
        if as_strategy === :typed
            return true, as_strategy
        elseif as_strategy === :none || lowered in _REPARAM_FALSE_STRINGS
            return false, :none
        elseif lowered in _REPARAM_TRUE_STRINGS
            if strategy_hint === :none
                return false, :none
            end
            return true, strategy_hint === nothing ? :typed : strategy_hint
        else
            return false, :none
        end
    else
        flag = Bool(raw_flag)
        if !flag || strategy_hint === :none
            return false, :none
        end
        return true, strategy_hint === nothing ? :typed : strategy_hint
    end
end

function _parse_hyperparameter_sweep(raw_section)::OrderedDict{String, Vector{Any}}
    """Parse hyperparameter sweep configuration from JSON."""
    result = OrderedDict{String, Vector{Any}}()
    
    if raw_section === nothing
        return result
    end
    
    if raw_section isa AbstractDict
        for (param_path, values) in raw_section
            if param_path isa AbstractString && values isa AbstractVector
                result[String(param_path)] = Any[v for v in values]
            end
        end
    end
    
    return result
end

mutable struct HyperparameterSweepSettings <: AbstractSettings
    hyperparameters::OrderedDict{String, Vector{Any}}

    function HyperparameterSweepSettings(raw_section_any)
        hyperparams = _parse_hyperparameter_sweep(raw_section_any)
        
        # If no hyperparameters provided, use sensible defaults for sweep
        if isempty(hyperparams)
            hyperparams = OrderedDict(
                "optimization_settings.param_bound_scaling_level" => ["medium", "high"],
                "optimization_settings.optimizer_settings.sigma0" => [2.0, 8.0],
                "optimization_settings.optimizer_settings.population_size" => [100, 150],
            )
        end
        
        return new(hyperparams)
    end
end

"""
    OptimizationSettings

Settings for network parameter optimization.

# Fields
- `method::String`: Optimization method. **Currently only "CMAES" is supported.**
- `param_bound_scaling_level::String`: Parameter bounds scaling level (low, medium, high, ultra, empirical, unbounded).
- `empirical_bounds_table_path::Union{String, Nothing}`: Path to CSV with empirical parameter bounds.
- `empirical_lower_bound_column::String`: Column name for lower bound values (e.g., "5perc").
- `empirical_upper_bound_column::String`: Column name for upper bound values (e.g., "95perc").
- `n_restarts::Int64`: Number of optimization restarts.
- Additional fields for loss configuration, reparameterization, and hyperparameter sweeping.
"""
mutable struct OptimizationSettings <: AbstractSettings
    method::String
    param_bound_scaling_level::String
    empirical_bounds_table_path::Union{String, Nothing}
    empirical_lower_bound_column::String
    empirical_upper_bound_column::String
    save_optimization_history::Bool
    save_modeled_psd::Bool
    include_settings_in_results_output::Bool
    reparametrize::Bool
    reparam_strategy::Symbol
    reparam_type_scales::Dict{Symbol, Float64}
    n_restarts::Int64
    maxiters::Int64
    time_limit_minutes::Int64
    loss_settings::LossSettings
    optimizer_settings::OptimizerSettings
    hyperparameter_sweep::HyperparameterSweepSettings
    output_dir::Union{String, Nothing}  # Path to optimization job folder (e.g., optimization_1/)

    function OptimizationSettings(dict::Dict)::OptimizationSettings
        optdict = get(dict, "optimization_settings", Dict{String, Any}())
        method = String(get(optdict, "method", "CMAES"))
        
        # Validate that only CMAES is supported
        if method != "CMAES"
            error("Optimization method '$method' is not supported. Only 'CMAES' is currently available.")
        end
        
        get_section = key -> begin
            section = get(optdict, key, nothing)
            if section isa AbstractDict
                return Dict{String, Any}(section)
            end
            fallback = get(dict, key, Dict{String, Any}())
            return fallback isa AbstractDict ? Dict{String, Any}(fallback) : Dict{String, Any}()
        end
        loss_section = get_section("loss_settings")
        lossset = LossSettings(loss_section)
        optz = OptimizerSettings(get_section("optimizer_settings"))
        raw_reparam = get(optdict, "reparametrize", true)
        raw_strategy = get(optdict, "reparam_strategy", nothing)
        reparam_flag, reparam_strategy = _parse_reparam_inputs(raw_reparam, raw_strategy)

        raw_scale_dict = get(optdict, "reparam_type_scales", Dict{String, Any}())
        reparam_type_scales = Dict{Symbol, Float64}()
        if raw_scale_dict isa AbstractDict
            for (k_raw, v_raw) in raw_scale_dict
                try
                    sym = Symbol(lowercase(strip(String(k_raw))))
                    scale_val = max(Float64(v_raw), eps())
                    reparam_type_scales[sym] = scale_val
                catch err
                    vwarn(string("Skipping invalid reparam_type_scales entry: key=", k_raw, " value=", v_raw); level=2)
                end
            end
        end

        raw_bound_scaling_level = get(optdict, "param_bound_scaling_level", nothing)
        param_bound_scaling_level = raw_bound_scaling_level === nothing ? "medium" : String(raw_bound_scaling_level)
        
        # Get empirical bounds table path (defaults to grammars/empirical_parameter_values.csv)
        # Uses universal path separators via joinpath for cross-platform compatibility
        empirical_bounds_table_path = get(optdict, "empirical_bounds_table_path", joinpath("grammars", "empirical_parameter_values.csv"))
        if empirical_bounds_table_path !== nothing
            empirical_bounds_table_path = String(empirical_bounds_table_path)
        end
        
        # Get empirical bounds column names (defaults to 5th/95th percentiles)
        empirical_lower_bound_column = String(get(optdict, "empirical_lower_bound_column", "5perc"))
        empirical_upper_bound_column = String(get(optdict, "empirical_upper_bound_column", "95perc"))
        
        save_optimization_history = _losssettings_as_bool(get(optdict, "save_optimization_history", false), false)
        save_modeled_psd = _losssettings_as_bool(get(optdict, "save_modeled_psd", false), false)
        include_settings_in_results_output = _losssettings_as_bool(get(optdict, "include_settings_in_results_output", true), true)
        n_restarts = Int64(get(optdict, "n_restarts", 1))
        raw_sweep_section = begin
            section = get(optdict, "hyperparameter_sweep", nothing)
            if section === nothing
                section = get(dict, "hyperparameter_sweep", Dict{String, Any}())
            end
            section isa AbstractDict ? Dict{String, Any}(section) : Dict{String, Any}()
        end
        hyper = HyperparameterSweepSettings(raw_sweep_section)
        
        # Output directory for this optimization job (set to nothing initially, will be set by optimize_network or run_hyperparameter_sweep)
        output_dir = get(optdict, "output_dir", nothing)
        if output_dir !== nothing
            output_dir = String(output_dir)
        end

    new(
            method,
            param_bound_scaling_level,
            empirical_bounds_table_path,
            empirical_lower_bound_column,
            empirical_upper_bound_column,
            save_optimization_history,
            save_modeled_psd,
            include_settings_in_results_output,
            reparam_flag,
            reparam_strategy,
            reparam_type_scales,
            n_restarts,
            Int64(get(optdict, "maxiters", 100_000)),
            Int64(get(optdict, "time_limit_minutes", 120)),
            lossset,
            optz,
            hyper,
            output_dir
        )
    end
end

function _normalize_symbol(val)::Symbol
    return Symbol(lowercase(strip(String(val))))
end

function _lowercase_dict(dict::AbstractDict)
    lowered = Dict{String, Any}()
    for (k, v) in dict
        lowered[lowercase(strip(String(k)))] = v
    end
    return lowered
end

function _convert_scale_sets(raw_sets)::Vector{Dict{Symbol, Float64}}
    raw_iter = raw_sets isa AbstractVector ? raw_sets : [raw_sets]
    converted = Dict{Symbol, Float64}[]
    for entry in raw_iter
        entry isa AbstractDict || continue
        dict = Dict{Symbol, Float64}()
        for (k, v) in entry
            dict[_normalize_symbol(k)] = Float64(v)
        end
        push!(converted, dict)
    end
    return converted
end

function _expand_scale_combinations(base::Dict{Symbol, Float64}, targets::Vector{Symbol}, values::Vector{Float64})
    isempty(targets) && return [deepcopy(base)]
    combos = Dict{Symbol, Float64}[Dict{Symbol, Float64}()]
    for sym in targets
        next = Dict{Symbol, Float64}[]
        for combo in combos, val in values
            dict = copy(combo)
            dict[sym] = val
            push!(next, dict)
        end
        combos = next
    end
    isempty(combos) && return [deepcopy(base)]
    return [merge(deepcopy(base), combo) for combo in combos]
end

function _parse_int_list(lowered::Dict{String, Any}, keys::Vector{String}; min_value::Int=1)
    for key in keys
        if haskey(lowered, key)
            values = Int[]
            raw_vals = lowered[key]
            iter_vals = raw_vals isa AbstractVector ? raw_vals : [raw_vals]
            for val in iter_vals
                v = Int(round(val))
                v >= min_value && push!(values, v)
            end
            !isempty(values) && return values, true
        end
    end
    return Int[], false
end

function _parse_float_list(lowered::Dict{String, Any}, keys::Vector{String}; min_value::Float64=eps())
    for key in keys
        if haskey(lowered, key)
            values = Float64[]
            raw_vals = lowered[key]
            iter_vals = raw_vals isa AbstractVector ? raw_vals : [raw_vals]
            for val in iter_vals
                v = Float64(val)
                v > min_value && push!(values, v)
            end
            !isempty(values) && return values, true
        end
    end
    return Float64[], false
end

_normalize_range_level(val) = begin
    s = lowercase(strip(String(val)))
    s == "large" && return "high"
    return s
end

function _parse_param_bound_scaling_levels(lowered::Dict{String, Any}, default_level::String)
    haskey(lowered, "param_bound_scaling_levels") || return [default_level]
    raw_vals = lowered["param_bound_scaling_levels"]
    iter_vals = raw_vals isa AbstractVector ? raw_vals : [raw_vals]
    levels = String[]
    for val in iter_vals
        label = _normalize_range_level(val)
        isempty(label) || push!(levels, label)
    end
    isempty(levels) && return [default_level]
    return unique(levels)
end

function _parse_sigma_mode(lowered::Dict{String, Any})::Symbol
    mode = lowercase(strip(String(get(lowered, "sigma0_mode", "auto"))))
    return mode in ("abs", "absolute") ? :absolute : :auto
end

function _build_scale_sets(lowered::Dict{String, Any}, base::Dict{Symbol, Float64})
    raw_base = isempty(base) ? Dict{Symbol, Float64}() : deepcopy(base)
    for key in ("reparam_scale_sets", "scale_sets")
        if haskey(lowered, key)
            sets = _convert_scale_sets(lowered[key])
            !isempty(sets) && return sets
        end
    end
    if haskey(lowered, "reparam_scale_values")
        vals_raw = lowered["reparam_scale_values"]
        vals_iter = vals_raw isa AbstractVector ? vals_raw : [vals_raw]
        vals = [Float64(val) for val in vals_iter]
        targets = get(lowered, "reparam_scale_targets", ["node_coupling", "population_coupling", "sensory_coupling"])
        syms = Symbol[_normalize_symbol(t) for t in targets]
        sets = _expand_scale_combinations(raw_base, syms, vals)
        !isempty(sets) && return sets
    end
    return [raw_base]
end

"""
    PSDSettings

Configuration for power spectral density (PSD) computation and preprocessing.

# Fields
- `window_size::Int`: Window size for Savitzky-Golay smoothing (default: 5)
- `smooth_poly_order::Int`: Polynomial order for Savitzky-Golay smoothing (default: 2)
- `rel_eps::Float64`: Relative epsilon for numerical stability (default: 1e-12)
- `smooth_sigma::Float64`: Gaussian smoothing sigma (default: 1.0)
- `workspace::Union{Nothing, SpectrumWorkspace}`: Cached workspace for Welch computation

These settings control the preprocessing applied after Welch PSD computation
and are independent of loss function configuration.
"""
mutable struct PSDSettings <: AbstractSettings
    preproc_pipeline::String                # Preprocessing pipeline: "log10", "offset", etc.
    welch_window_sec::Float64               # Welch window duration in seconds
    welch_overlap::Float64                  # Welch window overlap (0-1)
    welch_nperseg::Int                      # Welch segment length (0=auto)
    welch_nfft::Int                         # FFT length (0=auto)
    noise_avg_reps::Int                     # averaging reps for noisy PSD
    window_size::Int                        # Savitzky-Golay window size
    smooth_poly_order::Int                  # Savitzky-Golay polynomial order
    rel_eps::Float64                        # Relative epsilon for numerical stability
    smooth_sigma::Float64                   # Gaussian smoothing sigma
    transient_period_duration::Float64      # Duration (seconds) of initial transient to skip (0.0 to disable)
    noise_seed::Union{Int, Nothing}         # Random seed for PSD noise (42=deterministic, nothing=random)
    workspace::Union{Nothing, Any}          # Cached SpectrumWorkspace (using Any to avoid circular dep)

    function PSDSettings(dict::Dict{String, Any}=Dict{String, Any}())
        psd_dict = get(dict, "psd", Dict{String, Any}())
        
        preproc_pipeline = String(get(psd_dict, "preproc_pipeline", "log10"))
        welch_window_sec = max(Float64(get(psd_dict, "welch_window_sec", 2.0)), 0.0)
        welch_overlap = clamp(Float64(get(psd_dict, "welch_overlap", 0.1)), 0.0, 0.99)
        welch_nperseg = max(Int(get(psd_dict, "welch_nperseg", 0)), 0)
        welch_nfft = Int(get(psd_dict, "welch_nfft", 0))
        noise_avg_reps = max(Int(get(psd_dict, "noise_avg_reps", 1)), 1)
        
        window_size = max(Int(get(psd_dict, "window_size", 5)), 1)
        smooth_poly_order = max(Int(get(psd_dict, "smooth_poly_order", 2)), 0)
        rel_eps = Float64(get(psd_dict, "rel_eps", 1e-12))
        smooth_sigma = Float64(get(psd_dict, "smooth_sigma", 1.0))
        transient_period_duration = max(Float64(get(psd_dict, "transient_period_duration", 2.0)), 0.0)
        
        # Noise seed for deterministic vs random noise generation
        noise_seed = if haskey(psd_dict, "noise_seed")
            val = psd_dict["noise_seed"]
            val === nothing ? nothing : Int(val)
        else
            42  # Default: deterministic
        end
        
        return new(preproc_pipeline, welch_window_sec, welch_overlap, welch_nperseg, welch_nfft,
                   noise_avg_reps, window_size, smooth_poly_order, rel_eps, smooth_sigma, transient_period_duration, noise_seed, nothing)
    end
end

"""
    DataSettings

Configuration for data input and metadata.

# Fields
- `data_file`: Path to data CSV file
- `target_channel`: Name of target EEG channel
- `task_type`: Task associated with data
- `fs`: Sampling frequency (Hz)
- `data_columns`: Specific columns to load
- `estimate_measurement_noise`: Whether to estimate noise from data
- `spectral_roi_definition_mode`: Mode for defining region of interest (:auto or :manual)
- `spectral_roi_auto_peak_sensitivity`: Sensitivity for automatic peak detection (0.0-1.0)
- `spectral_roi_manual`: Manual frequency bands for ROI definition
- `psd`: Nested PSDSettings for spectrum computation

Note: DataSettings can be `nothing` if no data is used (e.g., for pure network building).
"""
mutable struct DataSettings <: AbstractSettings
    data_file::Union{String, Nothing}
    target_channel::Union{String, Dict{String, String}, Nothing}
    task_type::Union{String, Nothing}
    fs::Union{Float64, Nothing}
    data_columns::Union{Vector{String}, Nothing}
    estimate_measurement_noise::Bool
    spectral_roi_definition_mode::Symbol  # :auto or :manual
    spectral_roi_auto_peak_sensitivity::Float64  # 0.0-1.0, higher=looser peak detection
    spectral_roi_manual::Vector{Tuple{Float64, Float64}}  # [(fmin, fmax), ...] for manual mode
    measurement_noise_std::Float64  # Measurement noise standard deviation (0=no noise, -1.0=not estimated)
    psd::PSDSettings  # Nested PSD preprocessing settings (includes noise_seed)

    function DataSettings(dict::Dict{String, Any})::DataSettings
        # Handle data_file: can be absolute or relative path
        data_file = haskey(dict, "data_file") ? (dict["data_file"] === nothing ? nothing : String(dict["data_file"])) : nothing
        
        # If no data_file specified, try to use default example data file
        if data_file === nothing
            # pathof(ENEEGMA) returns /path/to/ENEEGMA/src/ENEEGMA.jl
            # We want /path/to/ENEEGMA/examples/
            eneegma_root = dirname(dirname(pathof(ENEEGMA)))
            examples_path = joinpath(eneegma_root, "examples")
            default_data_file = joinpath(examples_path, "example_data_rest.csv")
            if isfile(default_data_file)
                data_file = default_data_file
            end
        end
        
        # Resolve relative paths to examples folder (cross-platform)
        if data_file !== nothing && !isfile(data_file) && !isabspath(data_file)
            eneegma_root = dirname(dirname(pathof(ENEEGMA)))
            examples_path = joinpath(eneegma_root, "examples")
            candidate_path = joinpath(examples_path, basename(data_file))
            if isfile(candidate_path)
                data_file = candidate_path
            end
        end
        
        target_channel_raw = get(dict, "target_channel", "IC3")
        # Handle both String (single-node) and Dict (multi-node) formats
        target_channel = if target_channel_raw === nothing
            nothing
        elseif target_channel_raw isa AbstractDict
            # Multi-node: Dict{String, String} mapping node names to channels
            Dict{String, String}(target_channel_raw)
        else
            # Single-node: String channel name
            String(target_channel_raw)
        end
        
        task_type = haskey(dict, "task_type") ? (dict["task_type"] === nothing ? nothing : String(dict["task_type"])) : "rest"
        fs = haskey(dict, "fs") ? (dict["fs"] === nothing ? nothing : Float64(dict["fs"])) : 256.0
        data_columns = haskey(dict, "data_columns") ? (dict["data_columns"] === nothing ? nothing : Vector{String}(dict["data_columns"])) : nothing
        estimate_measurement_noise = _losssettings_as_bool(get(dict, "estimate_measurement_noise", true), true)
        
        # Region definition settings
        region_mode_raw = get(dict, "spectral_roi_definition_mode", "manual")
        spectral_roi_definition_mode = Symbol(lowercase(String(region_mode_raw)))
        spectral_roi_definition_mode in (:auto, :manual) || (spectral_roi_definition_mode = :auto)
        
        spectral_roi_auto_peak_sensitivity = clamp(Float64(get(dict, "spectral_roi_auto_peak_sensitivity", 0.3)), 0.0, 1.0)
        
        spectral_roi_manual = get(dict, "spectral_roi_manual", [[7.5, 14.0]])
        if spectral_roi_manual isa Vector
            try
                spectral_roi_manual = Vector{Tuple{Float64, Float64}}(
                    [begin
                        if r isa AbstractVector && length(r) >= 2
                            (Float64(r[1]), Float64(r[2]))
                        else
                            throw(ArgumentError("Invalid ROI format: $r"))
                        end
                    end for r in spectral_roi_manual]
                )
            catch e
                throw(ArgumentError("DataSettings: 'spectral_roi_manual' must be an array of 2-element arrays, " *
                    "e.g., [[4.0, 8.0], [12.0, 20.0]]. Got: $spectral_roi_manual. " *
                    "Note: JSON tuples (4.0, 8.0) are NOT valid—use [4.0, 8.0] instead. Error: $e"))
            end
        else
            spectral_roi_manual = Tuple{Float64, Float64}[]
        end
        
        # Initialize PSD settings
        psd = PSDSettings(dict)
        
        # Measurement noise standard deviation 
        measurement_noise_std = Float64(get(dict, "measurement_noise_std", 0.0))

        return new(data_file, target_channel, task_type, fs, data_columns, estimate_measurement_noise,
                   spectral_roi_definition_mode, spectral_roi_auto_peak_sensitivity, spectral_roi_manual, measurement_noise_std, psd)
    end
end

"""
    Settings

Complete configuration for ENEEGMA network building, simulation, and optimization.

Contains all subsettings organized by functional domain: general experiment settings,
network topology, sampling, simulation, data processing, and optimization.

# Fields
- `general_settings::GeneralSettings`: Experiment and output configuration.
- `network_settings::NetworkSettings`: Network topology and dynamics.
- `sampling_settings::Union{SamplingSettings, Nothing}`: Grammar-based sampling configuration.
- `simulation_settings::Union{SimulationSettings, Nothing}`: ODE solver configuration.
- `data_settings::Union{DataSettings, Nothing}`: Data input and preprocessing.
- `optimization_settings::Union{OptimizationSettings, Nothing}`: Parameter optimization configuration.
"""
mutable struct Settings
    general_settings::GeneralSettings
    network_settings::NetworkSettings
    sampling_settings::Union{SamplingSettings, Nothing}
    simulation_settings::Union{SimulationSettings, Nothing}
    data_settings::Union{DataSettings, Nothing}
    optimization_settings::Union{OptimizationSettings, Nothing}

    function Settings(dict::Dict{String, Any})

        gen  = GeneralSettings(dict)
        net  = NetworkSettings(dict)
        samp = SamplingSettings(dict)
        sim  = SimulationSettings(dict)
        opt  = OptimizationSettings(dict)
        datad = get(dict, "data_settings", Dict{String, Any}())
        data  = DataSettings(datad)

        return new(gen, net, samp, sim, data, opt)
    end
end

"""
Custom display for Settings objects - shows full configuration summary.
When displaying a Settings object in the REPL, print the full summary automatically.
"""
function Base.show(io::IO, ::MIME"text/plain", settings::Settings)
    print(io, "Settings object\n")
    print_settings_summary(settings; section="all")
end
