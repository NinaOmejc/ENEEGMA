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
        
        # Get network name with explicit default
        name = String(get(netsett, "name", "example-net"))
        
        # n_nodes is required or defaults to 1
        n_nodes = Int64(get(netsett, "n_nodes", 1))
        n_nodes > 0 || throw(ArgumentError("n_nodes must be > 0, got $n_nodes"))
        
        # Get or create node names - must match n_nodes
        node_names_raw = get(netsett, "node_names", nothing)
        node_names = if isnothing(node_names_raw) || isempty(node_names_raw)
            ["N$i" for i in 1:n_nodes]
        else
            nn = Vector{String}(node_names_raw)
            length(nn) == n_nodes || throw(ArgumentError("node_names length ($(length(nn))) must match n_nodes ($n_nodes)"))
            nn
        end
        
        # Get or create node models - must match n_nodes (supports both String and RuleTree)
        node_models_raw = get(netsett, "node_models", nothing)
        node_models = if isnothing(node_models_raw) || isempty(node_models_raw)
            fill("WC", n_nodes)
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
                Vector{String}(node_models_raw)
            else
                # Mixed or all RuleTree: keep as Vector{Union{String, RuleTree}}
                Vector{Union{String, RuleTree}}(node_models_raw)
            end
            length(nm) == n_nodes || throw(ArgumentError("node_models length ($(length(nm))) must match n_nodes ($n_nodes)"))
            nm
        end
        
        # Get or create node coordinates - must match n_nodes
        node_coords_raw = get(netsett, "node_coords", nothing)
        node_coords = if isnothing(node_coords_raw) || isempty(node_coords_raw)
            [(0.0, float(i)*10.0, 0.0) for i in 1:n_nodes]
        else
            nc = [(Float64(c[1]), Float64(c[2]), Float64(c[3])) for c in node_coords_raw]
            length(nc) == n_nodes || throw(ArgumentError("node_coords length ($(length(nc))) must match n_nodes ($n_nodes)"))
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
            (size(nc_mat) == (n_nodes, n_nodes)) || throw(ArgumentError("network_conn must be ($n_nodes × $n_nodes), got $(size(nc_mat))"))
            nc_mat
        end
        
        # Get or create connection functions matrix - must be n_nodes × n_nodes
        network_conn_funcs_raw = get(netsett, "network_conn_funcs", nothing)
        network_conn_funcs = if isnothing(network_conn_funcs_raw) || isempty(network_conn_funcs_raw)
            fill("", n_nodes, n_nodes)
        else
            ncf = if network_conn_funcs_raw isa AbstractDict
                parsed = _dict_to_matrix(network_conn_funcs_raw, String)
                parsed === nothing ? fill("", n_nodes, n_nodes) : parsed
            else
                permutedims(Matrix{String}(hcat(network_conn_funcs_raw...)))
            end
            (size(ncf) == (n_nodes, n_nodes)) || throw(ArgumentError("network_conn_funcs must be ($n_nodes × $n_nodes), got $(size(ncf))"))
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
            (size(nd_mat) == (n_nodes, n_nodes)) || throw(ArgumentError("network_delay must be ($n_nodes × $n_nodes), got $(size(nd_mat))"))
            nd_mat
        end
        
        # Get or create sensory input connectivity - must match n_nodes
        sensory_input_conn = let v = get(netsett, "sensory_input_conn", nothing)
            if isnothing(v) || isempty(v)
                ones(Int, n_nodes)  # Default: all nodes receive input
            else
                sic = Int.(v)
                length(sic) == n_nodes || throw(ArgumentError("sensory_input_conn length ($(length(sic))) must match n_nodes ($n_nodes)"))
                sic
            end
        end

        # Get sensory input function
        sensory_input_func = String(get(netsett, "sensory_input_func", "rand(Normal(0.0, 1.0))"))

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
        eeg_output = String(get(netsett, "eeg_output", ""))

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

mutable struct SamplingSettings <: AbstractSettings
    grammar_file::String
    n_samples::Int        
    only_unique::Bool
    max_resample_attempts::Int
    grammar_seed::Union{Int, Nothing}

    function SamplingSettings(dict::Dict{String, Any})::SamplingSettings
        sampdict = get(dict, "sampling_settings", Dict{String, Any}())
        
        # Construct grammar_file from path + filename (or use combined path if provided)
        grammar_file_raw = get(sampdict, "grammar_file", "grammars/default_grammar.cfg")
        grammar_file = grammar_file_raw === nothing || grammar_file_raw == "" ? "grammars/default_grammar.cfg" : String(grammar_file_raw)

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
            Int(get(   sampdict, "max_resample_attempts", 100)),
            grammar_seed
        )
    end
end


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
            Tuple{Float64, Float64}(get(simdict, "tspan", (0.0, 10.0))),
            Float64(get(simdict, "dt", 0.001)),
            Float64(get(simdict, "saveat", 0.001)),
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
            Int64(get(dict, "population_size", 100)),
            Float64(get(dict, "sigma0", 8.0))
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

    noise = _losssettings_ensure_dict(get(lossdict, "time_noise", nothing))
    if noise !== nothing
        if haskey(noise, "measurement_noise_std")
            lossdict["measurement_noise_std"] = Float64(noise["measurement_noise_std"])
        end
        if haskey(noise, "loss_noise_seed")
            raw_seed = noise["loss_noise_seed"]
            if raw_seed === nothing || raw_seed == ""
                lossdict["loss_noise_seed"] = nothing
            else
                lossdict["loss_noise_seed"] = _coerce_settings_idx_value(raw_seed)
            end
        end
    end

    return lossdict
end

mutable struct LossSettings <: AbstractSettings
    # PSD computation settings
    psd_preproc::String                    # preprocessing pipeline: "log10", "relative", etc.
    psd_welch_window_sec::Float64          # Welch window duration in seconds
    psd_welch_overlap::Float64             # Welch window overlap (0-1)
    psd_welch_nperseg::Int                 # Welch segment length (0=auto)
    psd_welch_nfft::Int                    # FFT length (0=auto)
    psd_noise_avg_reps::Int                # averaging reps for noisy PSD
    
    # Frequency analysis range
    fmin::Float64                          # minimum frequency (Hz)
    fmax::Float64                          # maximum frequency (Hz)
    
    # Noise modeling
    measurement_noise_std::Float64         # estimated standard deviation
    loss_noise_seed::Union{Nothing, Int}   # for reproducible noise
    
    # Region weighting (for roi vs background loss)
    roi_weight::Float64                    # weight for region of interest
    bg_weight::Float64                     # weight for background

    function LossSettings(dict::Dict{String, Any})::LossSettings
        cooked = Dict{String, Any}(dict)
        normalize_loss_settings_dict!(cooked)
        dict = cooked
        
        psd_welch_window_sec = max(Float64(get(dict, "psd_welch_window_sec", 2.0)), 0.0)
        psd_welch_overlap = clamp(Float64(get(dict, "psd_welch_overlap", 0.5)), 0.0, 0.99)
        psd_welch_nperseg = max(Int(get(dict, "psd_welch_nperseg", 0)), 0)
        psd_welch_nfft = Int(get(dict, "psd_welch_nfft", 0))
        psd_noise_avg_reps = max(Int(get(dict, "psd_noise_avg_reps", 1)), 1)

        raw_preproc = get(dict, "psd_preproc", nothing)
        psd_preproc = _canonicalize_psd_preproc_string(raw_preproc === nothing ? "log10" : String(raw_preproc))

        fmin = Float64(get(dict, "fmin", 1.0))
        fmax = Float64(get(dict, "fmax", 48.0))
        
        measurement_noise_std = max(Float64(get(dict, "measurement_noise_std", 0.0)), 0.0)

        # Default to deterministic seed (42) for reproducible results
        # Set explicitly to nothing to enable non-deterministic noise
        loss_noise_seed_val = get(dict, "loss_noise_seed", 42)
        loss_noise_seed = loss_noise_seed_val === nothing ? nothing : _coerce_settings_idx_value(loss_noise_seed_val)

        roi_weight = max(Float64(get(dict, "roi_weight", 1.0)), 0.0)
        bg_weight = max(Float64(get(dict, "bg_weight", 1.0)), 0.0)

        new(psd_preproc,
            psd_welch_window_sec,
            psd_welch_overlap,
            psd_welch_nperseg,
            psd_welch_nfft,
            psd_noise_avg_reps,
            fmin,
            fmax,
            measurement_noise_std,
            loss_noise_seed,
            roi_weight,
            bg_weight)
    end
end

function losssettings_psd_flags(ls::LossSettings)
    return _psd_preproc_tokens_flags(_psd_preproc_tokens(ls.psd_preproc))
end

function losssettings_psd_has_log(ls::LossSettings)
    has_log = losssettings_psd_flags(ls)
    return has_log
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

struct HyperparameterAxis
    hyperparameter::Vector{String}
    values::Vector{Any}
end

function _parse_hyperparameter_axes(raw_section)::Vector{HyperparameterAxis}
    axes = HyperparameterAxis[]
    raw_axes = get(raw_section, "hyperparameter_axes", nothing)
    raw_axes === nothing && return axes
    entries = raw_axes isa AbstractVector ? raw_axes : [raw_axes]
    for entry in entries
        entry isa AbstractDict || continue
        raw_keys = get(entry, "hyperparameter", nothing)
        raw_vals = get(entry, "values", nothing)
        raw_keys === nothing && continue
        raw_vals === nothing && continue
        keys = raw_keys isa AbstractVector ? [String(k) for k in raw_keys] : [String(raw_keys)]
        values_iter = raw_vals isa AbstractVector ? raw_vals : [raw_vals]
        values = Any[value for value in values_iter]
        isempty(keys) && continue
        isempty(values) && continue
        push!(axes, HyperparameterAxis(keys, values))
    end
    return axes
end

mutable struct HyperparameterSweepSettings <: AbstractSettings
    sigma_mode::Symbol
    param_range_levels::Vector{String}
    scale_sets::Vector{Dict{Symbol, Float64}}
    population_grid::Vector{Int}
    sigma_values_override::Union{Nothing, Vector{Float64}}
    restart_grid::Vector{Int}
    base_reparam_scales::Dict{Symbol, Float64}
    hyperparameter_axes::Vector{HyperparameterAxis}

    function HyperparameterSweepSettings(raw_section_any,
                                         param_range_level::String,
                                         reparam_type_scales::Dict{Symbol, Float64},
                                         n_restarts::Int,
                                         opt_settings::OptimizerSettings)
        raw_section = raw_section_any isa AbstractDict ? Dict{String, Any}(raw_section_any) : Dict{String, Any}()
        lowered = _lowercase_dict(raw_section)
        sigma_mode = _parse_sigma_mode(lowered)
        base_scales = isempty(reparam_type_scales) ? Dict{Symbol, Float64}() : deepcopy(reparam_type_scales)
        scale_sets = _build_scale_sets(lowered, base_scales)
        range_levels = _parse_param_range_levels(lowered, _normalize_range_level(param_range_level))
        pop_values, pop_found = _parse_int_list(lowered, ["population_sizes", "population_grid", "cmaes_population"]; min_value=1)
        population_grid = pop_found ? pop_values : [40, 80]
        restart_values, restart_found = _parse_int_list(lowered, ["n_restarts", "restart_counts", "restart_grid"]; min_value=1)
        restart_fallback = n_restarts > 0 ? n_restarts : 1
        restart_grid = restart_found ? restart_values : [restart_fallback]
        sigma_values, sigma_found = _parse_float_list(lowered, ["sigma0_values", "sigma0", "sigma_grid"]; min_value=eps())
        sigma_override = sigma_found ? sigma_values : [2.0, 8.0]
        axes = _parse_hyperparameter_axes(raw_section)
        return new(sigma_mode,
                   range_levels,
                   scale_sets,
                   population_grid,
                   sigma_override,
                   restart_grid,
                   base_scales,
                   axes)
    end
end

"""
    OptimizationSettings

Settings for network parameter optimization.

# Fields
- `method::String`: Optimization method. **Currently only "CMAES" is supported.**
- `n_restarts::Int64`: Number of optimization restarts
- Additional fields for loss configuration, reparameterization, and hyperparameter sweeping
"""
mutable struct OptimizationSettings <: AbstractSettings
    method::String
    loss_abstol::Float64
    loss_reltol::Float64
    abs_target_loss::Float64
    param_range_level::String
    empirical_param_table_path::Union{String, Nothing}
    empirical_lb_col::String
    empirical_ub_col::String
    save_optimization_history::Bool
    save_modeled_psd::Bool
    reparametrize::Bool
    reparam_strategy::Symbol
    reparam_type_scales::Dict{Symbol, Float64}
    n_restarts::Int64
    maxiters::Int64
    time_limit_minutes::Int64
    loss_settings::LossSettings
    optimizer_settings::OptimizerSettings
    hyperparameter_sweep::HyperparameterSweepSettings

    function OptimizationSettings(dict::Dict)::OptimizationSettings
        optdict = get(dict, "optimization_settings", Dict{String, Any}())
        method = String(get(optdict, "method", get(optdict, "optimization_method", "CMAES")))
        
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
        loss_abstol = Float64(get(loss_section, "loss_abstol", get(optdict, "loss_abstol", get(optdict, "loss_limit", 1e-5))))
        loss_reltol = Float64(get(loss_section, "loss_reltol", get(optdict, "loss_reltol", get(optdict, "rel_diff_convergence", 1e-5))))
        abs_target_loss = max(Float64(get(optdict, "abs_target_loss", 0.01)), 0.0)
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

        raw_range_level = get(optdict, "param_range_level", nothing)
        param_range_level = raw_range_level === nothing ? "medium" : String(raw_range_level)
        
        # Get empirical parameter table path
        empirical_param_table_path = get(optdict, "empirical_param_table_path", nothing)
        if empirical_param_table_path !== nothing
            empirical_param_table_path = String(empirical_param_table_path)
            # Expand to absolute path if needed
            if !isabspath(empirical_param_table_path)
                vwarn("empirical_param_table_path is not an absolute path: $(empirical_param_table_path)"; level=2)
            end
        end
        
        # Get empirical bounds column names (defaults to 5th/95th percentiles)
        empirical_lb_col = String(get(optdict, "empirical_lb_col", "q1"))
        empirical_ub_col = String(get(optdict, "empirical_ub_col", "q3"))
        
        save_optimization_history = _losssettings_as_bool(get(optdict, "save_optimization_history", false), false)
        save_modeled_psd = _losssettings_as_bool(get(optdict, "save_modeled_psd", false), false)
        n_restarts = Int64(get(optdict, "n_restarts", 1))
        raw_sweep_section = begin
            section = get(optdict, "hyperparameter_sweep", nothing)
            if section === nothing
                section = get(dict, "hyperparameter_sweep", get(dict, "hyper_sweep", Dict{String, Any}()))
            end
            section isa AbstractDict ? Dict{String, Any}(section) : Dict{String, Any}()
        end
        hyper = HyperparameterSweepSettings(raw_sweep_section, param_range_level, reparam_type_scales, n_restarts, optz)

    new(
            method,
            loss_abstol,
            loss_reltol,
            abs_target_loss,
            param_range_level,
            empirical_param_table_path,
            empirical_lb_col,
            empirical_ub_col,
            save_optimization_history,
            save_modeled_psd,
            reparam_flag,
            reparam_strategy,
            reparam_type_scales,
            n_restarts,
            Int64(get(optdict, "maxiters", 10000)),
            Int64(get(optdict, "time_limit_minutes", get(optdict, "time_limit", 120))),
            lossset,
            optz,
            hyper
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

function _parse_param_range_levels(lowered::Dict{String, Any}, default_level::String)
    haskey(lowered, "param_range_levels") || return [default_level]
    raw_vals = lowered["param_range_levels"]
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
    DataSettings

Configuration for data input and metadata.

# Fields
- `path_data`: Path to data directory
- `subj`: Subject identifier (String or Int)
- `task`: Task name
- `event`: Event identifier
- `fs`: Sampling frequency (Hz)
- `n_ts`: Number of time series
- `ts_type`: Type of time series (e.g., "comp")
- `fname_data`: Name of the data file
- `fname_metadata`: Name of the metadata file

Note: DataSettings can be `nothing` if no data is used (e.g., for pure network building).
"""
mutable struct DataSettings <: AbstractSettings
    data_file::Union{String, Nothing}
    target_channel::Union{String, Nothing}
    task_type::Union{String, Nothing}
    fs::Union{Float64, Nothing}
    data_columns::Union{Vector{String}, Nothing}
    estimate_measurement_noise::Bool
    region_definition_mode::Symbol  # :auto or :manual
    auto_peak_sensitivity::Float64  # 0.0-1.0, higher=looser peak detection
    manual_frequency_regions::Vector{Tuple{Float64, Float64}}  # [(fmin, fmax), ...] for manual mode

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
        
        target_channel = haskey(dict, "target_channel") ? (dict["target_channel"] === nothing ? nothing : String(dict["target_channel"])) : "IC3"
        task_type = haskey(dict, "task_type") ? (dict["task_type"] === nothing ? nothing : String(dict["task_type"])) : "rest"
        fs = haskey(dict, "fs") ? (dict["fs"] === nothing ? nothing : Float64(dict["fs"])) : 256.0
        data_columns = haskey(dict, "data_columns") ? (dict["data_columns"] === nothing ? nothing : Vector{String}(dict["data_columns"])) : nothing
        estimate_measurement_noise = _losssettings_as_bool(get(dict, "estimate_measurement_noise", true), true)
        
        # Region definition settings
        region_mode_raw = get(dict, "region_definition_mode", "auto")
        region_definition_mode = Symbol(lowercase(String(region_mode_raw)))
        region_definition_mode in (:auto, :manual) || (region_definition_mode = :auto)
        
        auto_peak_sensitivity = clamp(Float64(get(dict, "auto_peak_sensitivity", 0.3)), 0.0, 1.0)
        manual_frequency_regions = get(dict, "manual_frequency_regions", [])
        if manual_frequency_regions isa Vector
            manual_frequency_regions = Vector{Tuple{Float64, Float64}}(
                [(Float64(r[1]), Float64(r[2])) for r in manual_frequency_regions]
            )
        else
            manual_frequency_regions = Tuple{Float64, Float64}[]
        end

        return new(data_file, target_channel, task_type, fs, data_columns, estimate_measurement_noise,
                   region_definition_mode, auto_peak_sensitivity, manual_frequency_regions)
    end
end




mutable struct Settings
    general_settings::GeneralSettings
    network_settings::NetworkSettings
    sampling_settings::Union{SamplingSettings, Nothing}
    simulation_settings::Union{SimulationSettings, Nothing}
    optimization_settings::Union{OptimizationSettings, Nothing}
    data_settings::Union{DataSettings, Nothing}

    function Settings(dict::Dict{String, Any})

        gen  = GeneralSettings(dict)
        net  = NetworkSettings(dict)

        # Always construct sampling, simulation, optimization, and data with defaults if missing
        samp = SamplingSettings(dict)
        sim  = SimulationSettings(dict)
        opt  = OptimizationSettings(dict)

        datad = get(dict, "data_settings", Dict{String, Any}())
        data  = DataSettings(datad)

        return new(gen, net, samp, sim, opt, data)
    end
end
