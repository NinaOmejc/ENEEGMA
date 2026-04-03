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
        exp_name = String(get(gendict, "exp_name", "SimpleNetwork"))

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
            Bool(get(gendict, "make_plots", false)),
            verbosity_level,
            seed
        )
    end
end

mutable struct NetworkSettings <: AbstractSettings
    n_nodes::Int64
    node_names::Vector{String}
    node_models::Vector{String}
    node_coords::Vector{Tuple{Float64, Float64, Float64}}
    network_conn::Matrix{Float64}
    network_conn_funcs::Matrix{String}
    network_delay::Matrix{Float64}
    sensory_input_conn::Vector{Int64}
    sensory_input_func::String
    seed_sensory_input::Union{Int, Nothing}
    eeg_output::String

    # Constructor with validation built-in
    function NetworkSettings(settings::Dict{String, Any})::NetworkSettings
        # Get network settings dict
        netsett = get(settings, "network_settings", Dict{String, Any}())
        
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
        
        # Get or create node models - must match n_nodes
        node_models_raw = get(netsett, "node_models", nothing)
        node_models = if isnothing(node_models_raw) || isempty(node_models_raw)
            fill("WC", n_nodes)
        else
            nm = Vector{String}(node_models_raw)
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
        seed_sensory_input = get(netsett, "seed_sensory_input", nothing)
        if seed_sensory_input === nothing || seed_sensory_input == ""
            seed_sensory_input = nothing
        else
            seed_sensory_input = Int(seed_sensory_input)
        end

        # Get EEG output function
        eeg_output = String(get(netsett, "eeg_output", ""))

        return new(
            n_nodes,
            node_names,
            node_models,
            node_coords,
            network_conn,
            network_conn_funcs,
            network_delay,
            sensory_input_conn,
            sensory_input_func,
            seed_sensory_input,
            eeg_output
        )
    end
end

mutable struct SamplingSettings <: AbstractSettings
    grammar_file::String
    n_samples::Int        
    only_unique::Bool
    max_resample_attempts::Int
    seed::Union{Int, Nothing}

    function SamplingSettings(dict::Dict{String, Any})::SamplingSettings
        sampdict = get(dict, "sampling_settings", Dict{String, Any}())
        
        # Construct grammar_file from path + filename (or use combined path if provided)
        grammar_file_raw = get(sampdict, "grammar_file", nothing)
        grammar_file = if grammar_file_raw !== nothing && grammar_file_raw != ""
            String(grammar_file_raw)
        else
            # Fall back to constructing from path and filename
            path = String(get(sampdict, "path_grammar", "grammars"))
            fname = String(get(sampdict, "fname_grammar", "default_grammar.cfg"))
            joinpath(path, fname)
        end
        
        # Parse seed - allow null/empty to mean "random"
        seed_raw = get(sampdict, "seed", nothing)
        seed = if seed_raw === nothing || seed_raw == ""
            nothing
        else
            Int(seed_raw)
        end
        
        new(
            grammar_file,
            Int(get(   sampdict, "n_samples", 10)),
            Bool(get(  sampdict, "only_unique", true)),
            Int(get(   sampdict, "max_resample_attempts", 100)),
            seed
        )
    end
end


mutable struct SimulationSettings <: AbstractSettings
    new_params_map_subset::OrderedDict{String, Float64}
    new_params_range_subset::OrderedDict{String, Tuple{Float64, Float64}}
    new_inits_map_subset::OrderedDict{String, Float64}
    tspan::Tuple{Float64, Float64}
    n_runs::Int64
    dt::Union{Float64, Nothing}
    saveat::Float64
    abstol::Union{Float64, Nothing}
    reltol::Union{Float64, Nothing}
    solver::Union{String, Nothing}
    maxiters::Union{Int, Nothing}
    save_everystep::Bool
    force_resimulation::Bool
    verbose::Bool
    
    # Constructor with type conversion and default values
    function SimulationSettings(dict::Dict{String, Any})::SimulationSettings

        simdict = get(dict, "simulation_settings", Dict{String, Any}())

        # Process parameter maps
        params_map = OrderedDict{String, Float64}()
        for (key, value) in get(simdict, "new_params_map_subset", Dict())
            params_map[key] = Float64(value)
        end

        # Process parameter ranges
        params_range = OrderedDict{String, Tuple{Float64, Float64}}()
        for (key, value) in get(simdict, "new_params_range_subset", Dict())
            if value isa Vector && length(value) >= 2
                params_range[key] = (Float64(value[1]), Float64(value[2]))
            elseif value isa Tuple && length(value) == 2
                params_range[key] = (Float64(value[1]), Float64(value[2]))
            else
                @warn "Skipping invalid range for $key: $value. Expected [min, max] or (min, max)."
            end
        end

        # Process initial values
        inits_map = OrderedDict{String, Float64}()
        for (key, value) in get(simdict, "new_inits_map_subset", Dict())
            inits_map[key] = Float64(value)
        end

        new(
            params_map,
            params_range,
            inits_map,
            Tuple{Float64, Float64}(get(simdict, "tspan", (0.0, 10.0))),
            Int64(get(simdict, "n_runs", 1)),
            Float64(get(simdict, "dt", 0.001)),
            Float64(get(simdict, "saveat", 0.001)),
            get(simdict, "abstol", nothing),
            get(simdict, "reltol", nothing),
            String(get(simdict, "solver", "Tsit5")),
            Float64(get(simdict, "maxiters", -1)),
            Bool(get(simdict, "save_everystep", false)),
            Bool(get(simdict, "force_resimulation", false)),
            Bool(get(simdict, "verbose", false))
        )
    end
end


"""
    OptimizerSettings

Configuration specific to optimization algorithms.

# Fields
- `population_size`: Population size for CMAES
- `rel_diff_convergence`: Relative difference threshold for convergence

"""
mutable struct OptimizerSettings <: AbstractSettings
    population_size::Int64
    sigma0::Float64
    K::Float64
    n_samples::Int64
    learning_rate::Float64

    function OptimizerSettings(dict::Dict)

        new(
            Int64(get(dict, "population_size", 50)),
            Float64(get(dict, "sigma0", 1.0)),
            Float64(get(dict, "K", 0.5)),
            Int64(get(dict, "n_samples", 100)),
            Float64(get(dict, "learning_rate", 0.1)),
        )
    end
end

# ------------------------------------------------------------------
# PSD preprocessing helpers shared across loss settings and utilities
# ------------------------------------------------------------------

const _PSD_PREPROC_SPLIT_REGEX = r"[\,\s\-\|>]+"
const _LOSSSET_PSD_NORM_TOKENS = Set(["relative", "relative_db", "zscore", "minmax", "percent", "db"])
const _LOSSSET_PSD_LOG_TOKENS = Set(["log", "log10", "log2", "ln"])
const _LOSSSET_PSD_SMOOTH_TOKENS = Set(["savitzky_golay", "savitzkygolay", "savgol", "gaussian", "moving_avg", "movingavg", "ma"])

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
    has_norm = any(token -> token in _LOSSSET_PSD_NORM_TOKENS, tokens)
    has_log = any(token -> token in _LOSSSET_PSD_LOG_TOKENS, tokens)
    has_smooth = any(token -> token in _LOSSSET_PSD_SMOOTH_TOKENS, tokens)
    return has_norm, has_log, has_smooth
end

function _losssettings_ensure_dict(val)::Union{Dict{String, Any}, Nothing}
    val isa AbstractDict || return nothing
    return Dict{String, Any}(val)
end

function _losssettings_map_normalize_tail(tail::String)
    tail in ("relative_total", "relative", "rel", "relative_power") && return "relative"
    tail in ("relative_db", "reldb", "db") && return "relative_db"
    tail in ("zscore", "z", "z_score") && return "zscore"
    tail in ("minmax", "min_max") && return "minmax"
    tail in ("percent", "pct") && return "percent"
    tail in ("none", "off") && return "none"
    return tail
end

function _losssettings_map_smooth_tail(tail::String)
    tail in ("sgolay", "savitzky_golay", "savgol", "sgol") && return "savitzky_golay"
    tail in ("gauss", "gaussian") && return "gaussian"
    tail in ("ma", "movingavg", "moving_avg") && return "moving_avg"
    tail in ("none", "off") && return "none"
    return tail
end

function _losssettings_map_log_tail(tail::String)
    tail in ("10", "log10") && return "log10"
    tail in ("2", "log2") && return "log2"
    tail in ("e", "ln", "log", "natural") && return "log"
    return tail
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

function _losssettings_normalize_preproc_entry(entry)::Union{String, Nothing}
    token = lowercase(strip(String(entry)))
    isempty(token) && return nothing
    if occursin(":", token)
        parts = split(token, ":"; limit=2)
        head = strip(parts[1])
        tail = length(parts) > 1 ? strip(parts[2]) : ""
        if head in ("normalize", "norm")
            return _losssettings_map_normalize_tail(tail)
        elseif head in ("smooth", "smoothing", "filter")
            return _losssettings_map_smooth_tail(tail)
        elseif head in ("log", "logarithm")
            return _losssettings_map_log_tail(tail)
        end
        tail != "" && return tail
        return head
    end
    return token
end

function _losssettings_preproc_value_to_string(value)::Union{String, Nothing}
    value === nothing && return nothing
    if value isa AbstractVector
        @warn "psd.preproc no longer accepts arrays; please provide a single hyphen-separated string" value=value
        return nothing
    end
    spec = String(value)
    tokens = _psd_preproc_tokens(spec)
    mapped = String[]
    for entry in tokens
        mapped_token = _losssettings_normalize_preproc_entry(entry)
        mapped_token === nothing && continue
        push!(mapped, mapped_token)
    end
    isempty(mapped) && return spec
    return join(mapped, "-")
end

function normalize_loss_settings_dict!(lossdict::Dict{String, Any})
    psd = _losssettings_ensure_dict(get(lossdict, "psd", nothing))
    if psd !== nothing
        if haskey(psd, "fmin")
            lossdict["fmin"] = Float64(psd["fmin"])
        end
        if haskey(psd, "fmax")
            lossdict["fmax"] = Float64(psd["fmax"])
        end
        if haskey(psd, "fbands") && psd["fbands"] isa AbstractVector
            lossdict["fbands"] = String.(psd["fbands"])
        end
        preproc_val = _losssettings_preproc_value_to_string(get(psd, "preproc", nothing))
        preproc_val !== nothing && (lossdict["psd_preproc"] = preproc_val)
        normalize = _losssettings_ensure_dict(get(psd, "normalize", nothing))
        if normalize !== nothing && haskey(normalize, "eps")
            lossdict["psd_rel_eps"] = Float64(normalize["eps"])
        end
        smooth = _losssettings_ensure_dict(get(psd, "smooth", nothing))
        if smooth !== nothing
            haskey(smooth, "window_bins") && (lossdict["psd_window_size"] = Int(smooth["window_bins"]))
            haskey(smooth, "poly_order") && (lossdict["psd_poly_order"] = Int(smooth["poly_order"]))
            haskey(smooth, "sigma") && (lossdict["psd_smooth_sigma"] = Float64(smooth["sigma"]))
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

    peakbg = _losssettings_ensure_dict(get(lossdict, "peakbg", nothing))
    fspb = _losssettings_ensure_dict(get(lossdict, "fspb", nothing))
    if fspb !== nothing
        peak_det = _losssettings_ensure_dict(get(fspb, "peak_detection", nothing))
        if peak_det !== nothing
            lossdict["peak_detection_empty"] = isempty(peak_det)
            haskey(peak_det, "max_windows") && (lossdict["max_peak_windows"] = Int(peak_det["max_windows"]))
            haskey(peak_det, "bandwidth_hz") && (lossdict["peak_bandwidth_hz"] = Float64(peak_det["bandwidth_hz"]))
            haskey(peak_det, "prominence") && (lossdict["peak_prominence_db"] = Float64(peak_det["prominence"]))
            haskey(peak_det, "baseline_window_hz") && (lossdict["peak_baseline_window_hz"] = Float64(peak_det["baseline_window_hz"]))
            haskey(peak_det, "baseline_quantile") && (lossdict["peak_baseline_quantile"] = Float64(peak_det["baseline_quantile"]))
        else
            lossdict["peak_detection_empty"] = false
        end
        haskey(fspb, "min_frequency_hz") && (lossdict["peak_min_frequency_hz"] = Float64(fspb["min_frequency_hz"]))
        haskey(fspb, "max_frequency_hz") && (lossdict["peak_max_frequency_hz"] = Float64(fspb["max_frequency_hz"]))
        lossdict["fspb_enabled"] = _losssettings_as_bool(get(fspb, "enabled", true), true)
    end

    ssvep = _losssettings_ensure_dict(get(lossdict, "ssvep", nothing))
    if ssvep !== nothing
        lossdict["ssvep_enabled"] = _losssettings_as_bool(get(ssvep, "enabled", true), true)
        haskey(ssvep, "stim_freq_hz") && (lossdict["ssvep_stim_freq_hz"] = Float64(ssvep["stim_freq_hz"]))
        haskey(ssvep, "n_harmonics") && (lossdict["ssvep_n_harmonics"] = Int(ssvep["n_harmonics"]))
        haskey(ssvep, "bandwidth_hz") && (lossdict["ssvep_bandwidth_hz"] = Float64(ssvep["bandwidth_hz"]))
        haskey(ssvep, "harmonic_decay") && (lossdict["ssvep_harmonic_decay"] = Float64(ssvep["harmonic_decay"]))
    end

    if haskey(lossdict, "weight_fspb")
        lossdict["weight_fspb"] = Float64(lossdict["weight_fspb"])
    end
    if haskey(lossdict, "weight_ssvep")
        lossdict["weight_ssvep"] = Float64(lossdict["weight_ssvep"])
    end
    if haskey(lossdict, "weight_background")
        lossdict["weight_background"] = Float64(lossdict["weight_background"])
    end

    # _normalize_fspb_ssvep_weights_dict!(lossdict)

    noise = _losssettings_ensure_dict(get(lossdict, "time_noise", nothing))
    if noise !== nothing
        if haskey(noise, "sigma_meas")
            lossdict["sigma_meas"] = Float64(noise["sigma_meas"])
        end
        haskey(noise, "auto_initialize") && (lossdict["auto_initialize_sigma_meas"] = _losssettings_as_bool(noise["auto_initialize"], true))
        if haskey(noise, "noise_seed")
            raw_seed = noise["noise_seed"]
            if raw_seed === nothing || raw_seed == ""
                lossdict["noise_seed"] = nothing
            else
                lossdict["noise_seed"] = _coerce_settings_idx_value(raw_seed)
            end
        end
    end

    return lossdict
end

function _normalize_fspb_ssvep_weights_dict!(lossdict::Dict{String, Any})
    has_f = haskey(lossdict, "weight_fspb")
    has_s = haskey(lossdict, "weight_ssvep")
    has_b = haskey(lossdict, "weight_background")
    (has_f || has_s || has_b) || return

    wf = max(Float64(get(lossdict, "weight_fspb", 0.0)), 0.0)
    ws = max(Float64(get(lossdict, "weight_ssvep", 0.0)), 0.0)
    wb = if has_b
        max(Float64(get(lossdict, "weight_background", 0.0)), 0.0)
    elseif has_f && has_s
        max(1.0 - wf - ws, 0.0)
    else
        0.0
    end

    total = wf + ws + wb
    total > 0 || return

    lossdict["weight_fspb"] = wf / total
    lossdict["weight_ssvep"] = ws / total
    lossdict["weight_background"] = wb / total
    return nothing
end

mutable struct LossSettings <: AbstractSettings
    fmin::Float64
    fmax::Float64
    fbands::Vector{String}
    psd_preproc::String
    psd_window_size::Int
    psd_poly_order::Int
    psd_rel_eps::Float64
    psd_smooth_sigma::Float64
    psd_welch_window_sec::Float64
    psd_welch_overlap::Float64
    psd_welch_nperseg::Int
    psd_welch_nfft::Int
    psd_noise_avg_reps::Int
    sigma_meas::Float64
    auto_initialize_sigma_meas::Bool
    noise_seed::Union{Nothing, Int}
    peak_bandwidth_hz::Float64
    peak_prominence_db::Float64
    max_peak_windows::Int
    weight_background::Float64
    fspb_enabled::Bool
    peak_detection_empty::Bool
    peak_baseline_window_hz::Float64
    peak_baseline_quantile::Float64
    peak_min_frequency_hz::Float64
    peak_max_frequency_hz::Float64
    weight_fspb::Float64
    weight_ssvep::Float64
    ssvep_enabled::Bool
    ssvep_stim_freq_hz::Float64
    ssvep_n_harmonics::Int
    ssvep_bandwidth_hz::Float64
    ssvep_harmonic_decay::Float64
    max_abs_signal::Float64
    max_rms_growth::Float64
    psd_preproc_ops::Any
    psd_preproc_ops_cache_key::Any
    psd_workspace::Any

    function LossSettings(dict::Dict{String, Any})::LossSettings
        cooked = Dict{String, Any}(dict)
        normalize_loss_settings_dict!(cooked)
        dict = cooked
        fmin = Float64(get(dict, "fmin", 1.0))
        fmax = Float64(get(dict, "fmax", 48.0))
        fbands = Vector{String}(get(dict, "fbands", ["delta", "theta", "alpha", "betalow", "betahigh"]))
        psd_window_size = Int(get(dict, "psd_window_size", 5))
        psd_poly_order = Int(get(dict, "psd_poly_order", 2))
        psd_rel_eps = Float64(get(dict, "psd_rel_eps", 1e-12))
        psd_smooth_sigma = Float64(get(dict, "psd_smooth_sigma", 1.0))
        psd_welch_window_sec = max(Float64(get(dict, "psd_welch_window_sec", 2.0)), 0.0)
        psd_welch_overlap = clamp(Float64(get(dict, "psd_welch_overlap", 0.5)), 0.0, 0.99)
        psd_welch_nperseg = max(Int(get(dict, "psd_welch_nperseg", 0)), 0)
        psd_welch_nfft = Int(get(dict, "psd_welch_nfft", 0))
        psd_noise_avg_reps = max(Int(get(dict, "psd_noise_avg_reps", 1)), 1)

        raw_preproc = get(dict, "psd_preproc", nothing)
        psd_preproc = _canonicalize_psd_preproc_string(raw_preproc === nothing ? "log10" : String(raw_preproc))

        auto_initialize_sigma = _losssettings_as_bool(get(dict, "auto_initialize_sigma_meas", true), true)
        sigma_meas = max(Float64(get(dict, "sigma_meas", 0.0)), 0.0)

        noise_seed_val = get(dict, "noise_seed", nothing)
        noise_seed = noise_seed_val === nothing ? nothing : _coerce_settings_idx_value(noise_seed_val)

        peak_bandwidth_hz = max(Float64(get(dict, "peak_bandwidth_hz", 6.0)), eps())
        peak_prominence_db = Float64(get(dict, "peak_prominence_db", 0.5))
        max_peak_windows = max(Int(get(dict, "max_peak_windows", 2)), 0)
        weight_background = max(Float64(get(dict, "weight_background", 0.4)), 0.0)
        fspb_enabled = _losssettings_as_bool(get(dict, "fspb_enabled", true), true)
        peak_detection_empty = _losssettings_as_bool(get(dict, "peak_detection_empty", false), false)
        peak_baseline_window_hz = max(Float64(get(dict, "peak_baseline_window_hz", 6.0)), 0.5)
        peak_baseline_quantile = clamp(Float64(get(dict, "peak_baseline_quantile", 0.2)), 0.0, 1.0)
        peak_min_frequency_hz = max(Float64(get(dict, "peak_min_frequency_hz", 5.0)), 0.0)
        peak_max_frequency_hz = max(Float64(get(dict, "peak_max_frequency_hz", 45.0)), 0.0)
        weight_fspb = max(Float64(get(dict, "weight_fspb", 1.0)), 0.0)
        weight_ssvep = max(Float64(get(dict, "weight_ssvep", 1.0)), 0.0)
        ssvep_enabled = _losssettings_as_bool(get(dict, "ssvep_enabled", true), true)
        ssvep_stim_freq_hz = max(Float64(get(dict, "ssvep_stim_freq_hz", 5.0)), 0.0)
        ssvep_n_harmonics = max(Int(get(dict, "ssvep_n_harmonics", 3)), 1)
        ssvep_bandwidth_hz = max(Float64(get(dict, "ssvep_bandwidth_hz", 0.5)), 0.0)
        ssvep_harmonic_decay = max(Float64(get(dict, "ssvep_harmonic_decay", 0.7)), 0.0)
        max_abs_signal = max(Float64(get(dict, "max_abs_signal", 100.0)), 0.0)
        max_rms_growth = max(Float64(get(dict, "max_rms_growth", 100.0)), 0.0)

        psd_preproc_ops = nothing
        psd_preproc_ops_cache_key = nothing
        psd_workspace = nothing

        new(fmin,
            fmax,
            fbands,
            psd_preproc,
            psd_window_size,
            psd_poly_order,
            psd_rel_eps,
            psd_smooth_sigma,
            psd_welch_window_sec,
            psd_welch_overlap,
            psd_welch_nperseg,
            psd_welch_nfft,
            psd_noise_avg_reps,
            sigma_meas, auto_initialize_sigma, noise_seed,
            peak_bandwidth_hz, peak_prominence_db, max_peak_windows,
            weight_background, fspb_enabled, peak_detection_empty,
            peak_baseline_window_hz, peak_baseline_quantile, peak_min_frequency_hz, peak_max_frequency_hz,
            weight_fspb, weight_ssvep,
            ssvep_enabled, ssvep_stim_freq_hz, ssvep_n_harmonics, ssvep_bandwidth_hz,
            ssvep_harmonic_decay,
            max_abs_signal, max_rms_growth,
            psd_preproc_ops, psd_preproc_ops_cache_key,
            psd_workspace)
    end
end

function normalize_fspb_ssvep_weights!(ls::LossSettings; background_auto::Bool=false)
    wf = max(ls.weight_fspb, 0.0)
    ws = max(ls.weight_ssvep, 0.0)
    wb = max(ls.weight_background, 0.0)

    if background_auto
        wb = max(1.0 - wf - ws, 0.0)
        ls.weight_background = wb
    end

    total = wf + ws + wb
    total > 0 || return ls

    ls.weight_fspb = wf / total
    ls.weight_ssvep = ws / total
    ls.weight_background = wb / total
    return ls
end

function losssettings_psd_flags(ls::LossSettings)
    return _psd_preproc_tokens_flags(_psd_preproc_tokens(ls.psd_preproc))
end

function losssettings_psd_has_log(ls::LossSettings)
    _, has_log, _ = losssettings_psd_flags(ls)
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
    save_results::String

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
        pop_fallback = opt_settings.population_size > 0 ? opt_settings.population_size : max(opt_settings.λ, 32)
        pop_values, pop_found = _parse_int_list(lowered, ["population_sizes", "population_grid", "cmaes_population"]; min_value=1)
        population_grid = pop_found ? pop_values : [pop_fallback]
        restart_values, restart_found = _parse_int_list(lowered, ["n_restarts", "restart_counts", "restart_grid"]; min_value=1)
        restart_fallback = n_restarts > 0 ? n_restarts : 1
        restart_grid = restart_found ? restart_values : [restart_fallback]
        sigma_values, sigma_found = _parse_float_list(lowered, ["sigma0_values", "sigma0", "sigma_grid"]; min_value=eps())
        sigma_override = sigma_found ? sigma_values : nothing
        axes = _parse_hyperparameter_axes(raw_section)
        save_pref = lowercase(String(get(lowered, "save_results", "best")))
        save_pref in ("all", "best", "none") || (save_pref = "best")
        return new(sigma_mode,
                   range_levels,
                   scale_sets,
                   population_grid,
                   sigma_override,
                   restart_grid,
                   base_scales,
                   axes,
                   save_pref)
    end
end

"""
    OptimizationSettings

Settings for network parameter optimization.

# Fields
- `method::String`: Optimization method. **Currently only "CMAES" is supported.**
- `loss::String`: Loss function specification
- `n_restarts::Int64`: Number of optimization restarts
- Additional fields for loss configuration, reparameterization, and hyperparameter sweeping

Note: Previous optimization methods (DE, LBFGS, ADAM_NOGRAD) are no longer supported.
Only CMAES (Covariance Matrix Adaptation Evolution Strategy) is available.
"""
mutable struct OptimizationSettings <: AbstractSettings
    method::String
    loss::String
    loss_abstol::Float64
    loss_reltol::Float64
    abs_target_loss::Float64
    component_fit::String
    param_range_level::String
    empirical_param_table_path::Union{String, Nothing}
    empirical_lb_col::String
    empirical_ub_col::String
    save_optimization_history::Bool
    save_all_optim_restarts_results::Bool
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
        loss_name = String(get(loss_section, "loss", get(optdict, "loss", "fspb")))
#=         if lowercase(loss_name) in ("fspb+ssvep", "peakbg+ssvep", "fspb_ssvep", "fspb-ssvep")
            normalize_fspb_ssvep_weights!(lossset; background_auto=false)
        end =#
        loss_abstol = Float64(get(loss_section, "loss_abstol", get(optdict, "loss_abstol", get(optdict, "loss_limit", 1e-5))))
        loss_reltol = Float64(get(loss_section, "loss_reltol", get(optdict, "loss_reltol", get(optdict, "rel_diff_convergence", 1e-5))))
        abs_target_loss = max(Float64(get(optdict, "abs_target_loss", 0.01)), 0.0)
        component_fit = lowercase(String(get(optdict, "component_fit", "all")))
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
                    @warn "Skipping invalid reparam_type_scales entry" key=k_raw value=v_raw err=err
                end
            end
        end

        raw_range_level = get(optdict, "param_range_level", nothing)
        param_range_level = raw_range_level === nothing ? "high" : String(raw_range_level)
        
        # Get empirical parameter table path
        empirical_param_table_path = get(optdict, "empirical_param_table_path", nothing)
        if empirical_param_table_path !== nothing
            empirical_param_table_path = String(empirical_param_table_path)
            # Expand to absolute path if needed
            if !isabspath(empirical_param_table_path)
                @warn "empirical_param_table_path is not an absolute path: $(empirical_param_table_path)"
            end
        end
        
        # Get empirical bounds column names (defaults to 5th/95th percentiles)
        empirical_lb_col = String(get(optdict, "empirical_lb_col", "q1"))
        empirical_ub_col = String(get(optdict, "empirical_ub_col", "q3"))
        
        save_optimization_history = _losssettings_as_bool(get(optdict, "save_optimization_history", false), false)
        save_all_restarts = _losssettings_as_bool(get(optdict, "save_all_optim_restarts_results", true), true)
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
            loss_name,
            loss_abstol,
            loss_reltol,
            abs_target_loss,
            component_fit,
            param_range_level,
            empirical_param_table_path,
            empirical_lb_col,
            empirical_ub_col,
            save_optimization_history,
            save_all_restarts,
            save_modeled_psd,
            reparam_flag,
            reparam_strategy,
            reparam_type_scales,
            n_restarts,
            Int64(get(optdict, "maxiters", 2000)),
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
    data_path::Union{String, Nothing}
    data_fname::Union{String, Nothing}
    fs::Union{Float64, Nothing}
    data_columns::Union{Vector{String}, Nothing}
    target_channel::Union{String, Nothing}
    task_type::Union{String, Nothing}
    metadata_path::Union{String, Nothing}
    metadata_file::Union{String, Nothing}

    function DataSettings(dict::Dict{String, Any})::DataSettings
        data_path = haskey(dict, "data_path") ? (dict["data_path"] === nothing ? nothing : String(dict["data_path"])) : nothing
        data_fname = haskey(dict, "data_fname") ? (dict["data_fname"] === nothing ? nothing : String(dict["data_fname"])) : nothing
        fs = haskey(dict, "fs") ? (dict["fs"] === nothing ? nothing : Float64(dict["fs"])) : nothing
        data_columns = haskey(dict, "data_columns") ? (dict["data_columns"] === nothing ? nothing : Vector{String}(dict["data_columns"])) : nothing
        target_channel = haskey(dict, "target_channel") ? (dict["target_channel"] === nothing ? nothing : String(dict["target_channel"])) : nothing
        task_type = haskey(dict, "task_type") ? (dict["task_type"] === nothing ? nothing : String(dict["task_type"])) : nothing
        metadata_path = haskey(dict, "metadata_path") ? (dict["metadata_path"] === nothing ? nothing : String(dict["metadata_path"])) : nothing
        metadata_file = haskey(dict, "metadata_file") ? (dict["metadata_file"] === nothing ? nothing : String(dict["metadata_file"])) : nothing

        return new(data_path, data_fname, fs, data_columns, target_channel, task_type, metadata_path, metadata_file)
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

        # Always construct sampling, simulation, and optimization with defaults if missing
        samp = SamplingSettings(dict)
        sim  = SimulationSettings(dict)
        opt  = OptimizationSettings(dict)

        datad = get(dict, "data_settings", nothing)
        data  = datad === nothing ? nothing : DataSettings(datad)

        return new(gen, net, samp, sim, opt, data)
    end
end
