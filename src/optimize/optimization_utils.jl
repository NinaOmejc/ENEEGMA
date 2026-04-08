# ---------------------------------------------------------------------------
# Pre-sampled model recipe loading
# ---------------------------------------------------------------------------

struct OptimizationResults
    net_name::String
    exp_name::String
    task_type::String
    data_path::Union{Nothing, String}
    data_file::Union{Nothing, String}
    data_sub::Union{Nothing, String}
    data_ic::Union{Nothing, String}
    settings_idx::Union{Int, Nothing}
    settings_path::Union{String, Nothing}
    model_name::String
    node_models::String
    model_idx::Int
    hyperparam_idx::Union{Nothing, Int}
    restart_idx::Union{Nothing, Int}
    orig_idx::Int
    loss::Float64
    r2::Float64
    mae::Float64
    maef::Float64
    iae::Float64
    iaep::Float64
    iaeb::Float64
    is_best_combo_iae::Bool
    is_best_combo_iaep::Bool
    duration_seconds::Float64
    hyperparam_combo::Union{Nothing, Any}
    hyperparam_keys::Union{Nothing, Any}
    retcode::String
    best_params::Union{NamedTuple, Dict{String, Float64}}
    best_inits::Union{Dict{String, Float64}, Vector{Float64}}
    param_ranges::Dict{String, Tuple{Float64, Float64}}
    init_ranges::Dict{String, Tuple{Float64, Float64}}
    param_types::Dict{String, String}
end

function _coerce_float(val)
    if val isa AbstractString
        parsed = tryparse(Float64, strip(val))
        return parsed === nothing ? NaN : parsed
    elseif val isa Number
        return Float64(val)
    else
        return NaN
    end
end

function _coerce_int(val)
    if val isa Integer
        return Int(val)
    elseif val isa Real
        return Int(round(val))
    elseif val isa AbstractString
        parsed = tryparse(Int, strip(val))
        return parsed === nothing ? 0 : parsed
    else
        return 0
    end
end

function _dict_string_float(raw::AbstractDict)
    out = Dict{String, Float64}()
    for (k, v) in raw
        val = _coerce_float(v)
        isfinite(val) || continue
        out[String(k)] = val
    end
    return out
end

function _range_dict(entries)
    ranges = Dict{String, Tuple{Float64, Float64}}()
    types = Dict{String, String}()
    entries === nothing && return ranges, types
    for entry in entries
        entry isa AbstractDict || continue
        name = String(get(entry, "name", ""))
        isempty(name) && continue
        bounds = get(entry, "native_bounds", nothing)
        if bounds isa AbstractVector && length(bounds) >= 2
            lo = _coerce_float(bounds[1])
            hi = _coerce_float(bounds[2])
            if isfinite(lo) && isfinite(hi)
                ranges[name] = (lo, hi)
            end
        end
        if haskey(entry, "param_type")
            types[name] = String(get(entry, "param_type", ""))
        end
    end
    return ranges, types
end

function _inits_dict(init_ranges::Dict{String, Tuple{Float64, Float64}}, raw_inits)
    out = Dict{String, Float64}()
    raw_inits isa AbstractVector || return out
    names = collect(keys(init_ranges))
    if length(names) == length(raw_inits)
        for (name, val) in zip(names, raw_inits)
            v = _coerce_float(val)
            isfinite(v) || continue
            out[name] = v
        end
    end
    return out
end

function _translate_settings_path(path::AbstractString)
    raw = String(path)
    if startswith(raw, "/ceph/grid/home/nomejc/eneegma_exp/experiments/")
        m = match(r"^/ceph/grid/home/nomejc/eneegma_exp/experiments/([^/]+)/(.+)$", raw)
        if m !== nothing
            # Return path as-is (user should configure output paths in settings)
            return raw
        end
    end
    return raw
end

function _translate_data_path(path::AbstractString)
    raw = String(path)
    # Translate HPC data path to Windows path
    if startswith(raw, "/ceph/grid/home/nomejc")
        return get_local_data_path()  # Implement this function to return the local data path   
    end
    return raw
end

function get_brain_source_idx(net::Network)::Int
    brain_source_name = String(net.nodes[1].brain_source)
    state_syms = get_symbols(get_state_vars(net.vars); sort=true)
    state_syms_sym = Symbol.(state_syms)
    brain_source_idx = findfirst(==(Symbol(brain_source_name)), state_syms_sym)
    if brain_source_idx === nothing || brain_source_idx < 1
        return 1
    end
    return brain_source_idx
end

"""
    load_model_from_csv(models_path::String, model_idx::Int) -> String

Load a model recipe from a CSV file at the specified 1-based index.
Expects a CSV with a "recipe" column containing model specification strings.
"""
function load_model_from_csv(models_path::AbstractString, model_idx::Int)::DataFrameRow{DataFrame, DataFrames.Index}
    if !isfile(models_path)
        error("Models CSV file not found: $(models_path)")
    end
    
    recipes_df = CSV.read(models_path, DataFrame)
    
    if !hasproperty(recipes_df, :recipe)
        error("CSV file must contain a 'recipe' column: $(models_path)")
    end
    
    n_models = nrow(recipes_df)
    if model_idx < 1 || model_idx > n_models
        error("Model index $(model_idx) out of range [1, $(n_models)] in $(models_path)")
    end
    
    model = recipes_df[model_idx, :]
    return model
end

"""
    load_optimization_results(path::AbstractString; dicttype=Dict) -> AbstractDict

Load optimization results from a JSON file and return a dictionary.
"""
function load_optimization_results(path::AbstractString; dicttype=Dict)
    isfile(path) || error("Optimization results JSON not found: $(path)")
    raw = JSON.parsefile(path; dicttype=dicttype)

    net_name = String(get(raw, "net_name", ""))
    exp_name = String(get(raw, "exp_name", ""))
    settings_idx_val = get(raw, "settings_idx", nothing)
    settings_idx = settings_idx_val === nothing ? nothing : _coerce_int(settings_idx_val)
    settings_path_val = get(raw, "settings_path", nothing)
    settings_path_updated = settings_path_val === nothing ? nothing : _translate_settings_path(String(settings_path_val))
    node_models = String(get(raw, "node_models", ""))
    task_type = String(get(raw, "task_type", ""))
    hyperparam_idx = _coerce_int(get(raw, "hyperparam_idx", nothing))
    hyperparam_combo = get(raw, "hyperparam_combo", nothing)
    hyperparam_keys = get(raw, "hyperparam_keys", nothing)
    restart_idx = _coerce_int(get(raw, "restart_idx", nothing))
    orig_idx = _coerce_int(get(raw, "orig_idx", 0))
    loss = _coerce_float(get(raw, "loss", NaN))
    r2 = _coerce_float(get(raw, "r2", NaN))
    mae = _coerce_float(get(raw, "fsmae", NaN))
    maef = _coerce_float(get(raw, "maef", NaN))
    iae = _coerce_float(get(raw, "iae", NaN))
    iaep = _coerce_float(get(raw, "iaep", NaN))
    iaeb = _coerce_float(get(raw, "iaeb", NaN))
    is_best_combo_iae = get(raw, "is_best_combo_iae", false)
    is_best_combo_iaep = get(raw, "is_best_combo_iaep", false)
    duration_seconds = _coerce_float(get(raw, "duration_seconds", NaN))
    retcode = String(get(raw, "retcode", ""))
    model_name = String(get(raw, "model_name", node_models))
    model_idx = _coerce_int(get(raw, "model_idx", 0))

    best_params = get(raw, "best_params", Dict{String, Any}())
    best_params = _dict_string_float(best_params)
    #best_params = (; (Symbol(k) => v for (k, v) in best_params)...)

    if exp_name == "" && !isempty(settings_path_updated)
        # Try to infer exp_name from settings_path_updated
        m = match(r"(?:^|[\\/])(exp\d)(?:[\\/]|$)", settings_path_updated)
        if m !== nothing
            exp_name = m.captures[1]
        end
    end

    if hyperparam_combo isa AbstractVector && hyperparam_keys isa AbstractVector
        hyperparam_combo = tuple(hyperparam_combo...)
        hyperparam_keys = [String(k) for k in hyperparam_keys]
    end

    param_ranges, param_types = _range_dict(get(raw, "parameter_ranges", nothing))
    init_ranges, _ = _range_dict(get(raw, "initial_state_ranges", nothing))

    best_inits = get(raw, "initial_states", nothing)
    if best_inits isa AbstractVector
        best_inits = Vector{Float64}(best_inits)
    end

    data_path = get(raw, "data_path", nothing)
    if data_path isa AbstractString
        data_path = _translate_data_path(data_path)
    else
        data_path = get_local_data_path()
    end

    data_file = get(raw, "data_file", get(raw, "data_fname", nothing))
    data_sub = get(raw, "data_sub", nothing)
    data_ic = get(raw, "data_ic", nothing)

    if data_file isa AbstractString
        data_file = String(data_file)
        data_sub = String(data_sub)
        data_ic = String(data_ic)
    else
        # load settings file to infer data_file if not present in results
        if !isnothing(settings_path_updated) && isfile(settings_path_updated)
            settings_json = JSON.parsefile(settings_path_updated; dicttype=Dict)
            # data file is inside settings_json.data_settings
            data_settings = get(settings_json, "data_settings", Dict{String, Any}())
            data_file = String(get(data_settings, "data_file", ""))
            data_sub = data_file == "" ? "" : split(data_file, "_")[1]
            data_ic = String(get(data_settings, "target_channel", ""))
        else
            data_file = ""
        end
    end

    return OptimizationResults(
        net_name,
        exp_name,
        task_type,
        data_path,
        data_file,
        data_sub,
        data_ic,
        settings_idx,
        settings_path_updated,
        model_name,
        node_models,
        model_idx,
        hyperparam_idx,
        restart_idx,
        orig_idx,
        loss,
        r2,
        mae,
        maef,
        iae,
        iaep,
        iaeb,
        is_best_combo_iae,
        is_best_combo_iaep,
        duration_seconds,
        hyperparam_combo,
        hyperparam_keys,
        retcode,
        best_params,
        best_inits,
        param_ranges,
        init_ranges,
        param_types
    )
end

"""
    apply_recipe_to_settings!(network_settings::NetworkSettings, recipe::String, model_idx::Int)

Apply a model recipe string to network settings, updating node models and network name.
"""
function apply_model_to_settings!(settings::Settings, recipe::AbstractString, model_name::AbstractString, model_idx::Int)
    if model_name in list_known_node_models_codes()
        settings.network_settings.node_models = [model_name]
        vprint("Applied known node model code: $(model_name)"; level=2)
    else
        settings.network_settings.node_models = [recipe]
    end
    settings.general_settings.exp_name = "$(settings.general_settings.exp_name)_$(model_name)"
    settings.sampling_settings.model_idx = model_idx
    _rebuild_path_out!(settings)
    
    return settings
end

function _rebuild_path_out!(settings::Settings)
    gen = settings.general_settings
    net = settings.network_settings
    network_name = net.network_name
    settings_idx = gen.settings_idx

    current_path = gen.path_out
    path_parts = splitpath(current_path)
    if !isempty(path_parts) && occursin(r"^.+-s\d+$", path_parts[end])
        path_out_root = joinpath(path_parts[1:end-1]...)
    else
        path_out_root = dirname(current_path)
    end

    gen.path_out = joinpath(path_out_root, "$(network_name)-s$(settings_idx)")
    mkpath(gen.path_out)
    return settings
end

# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

"""
    get_indices(param_syms, all_params_tuple)

Return the indices of parameters in `all_params_tuple` that are present in
`param_syms`. Assumes `param_syms` is a collection of Symbols and
`all_params_tuple` is a NamedTuple.
"""
function get_indices(param_syms, all_params_tuple)
    param_names = propertynames(all_params_tuple)
    return [findfirst(==(s), param_names) for s in param_syms if s in param_names]
end


struct OptLogEntry
    irestart::Int64
    iter::Int64
    loss::Float64
    time::Second
    params::Union{Nothing, Vector{Float64}}  # tunable params, state inits, optional measurement noise
end


"""
    LossTimingTracker(limit=10)

Collect wall-clock durations (ms) for the first `limit` loss evaluations. Helps
compare different optimization configurations without affecting later runs.
"""
mutable struct LossTimingTracker
    limit::Int
    samples::Vector{Float64}
    eval_count::Int
    function LossTimingTracker(limit::Integer=10)
        limit = max(Int(limit), 0)
        return new(limit, Float64[], 0)
    end
end

"""
    track_loss_timing(lossfun, tracker)

Wrap `lossfun` so that the first `tracker.limit` evaluations record their
durations in milliseconds. Later evaluations bypass timing to avoid overhead.
"""
function track_loss_timing(lossfun::Function, tracker::LossTimingTracker)
    tracker.limit <= 0 && return lossfun
    return function(z, args)
        if length(tracker.samples) < tracker.limit
            start_ns = time_ns()
            val = lossfun(z, args)
            duration_ms = (time_ns() - start_ns) / 1.0e6
            push!(tracker.samples, duration_ms)
            tracker.eval_count += 1
            return val
        elseif length(tracker.samples) == tracker.limit
            tracker.eval_count += 1
            println("Loss timing tracker reached limit of $(tracker.limit) samples after $(tracker.eval_count) evaluations.")
            return lossfun(z, args)
        else
            tracker.eval_count += 1
            return lossfun(z, args)
        end
    end
end

"""
    summarize_loss_timing(tracker)

Return `(mean_ms, std_ms, n)` for recorded samples or `nothing` if empty.
"""
function summarize_loss_timing(tracker::LossTimingTracker)
    isempty(tracker.samples) && return nothing
    mean_val = mean(tracker.samples)
    std_val = length(tracker.samples) > 1 ? std(tracker.samples) : 0.0
    return (mean_ms=mean_val, std_ms=std_val, n=length(tracker.samples))
end


function snapshot_param_ranges(paramset::ParamSet)
    return [(p.min, p.max) for p in paramset.params]
end

function restore_param_ranges!(paramset::ParamSet, snapshot::Vector{Tuple{Float64, Float64}})
    length(paramset.params) == length(snapshot) || error("Range snapshot and ParamSet size mismatch.")
    @inbounds for i in eachindex(paramset.params)
        param = paramset.params[i]
        bounds = snapshot[i]
        param.min = bounds[1]
        param.max = bounds[2]
    end
    return paramset
end


## ------------------------------------------------------------------
## Population heuristics
## ------------------------------------------------------------------

const _DEFAULT_POP_MIN = 12
const _DEFAULT_POP_MAX = 128
const _DEFAULT_POP_SCALE = 2.0

"""
    infer_population_size(decision_dim; min_pop=12, max_pop=64, scale=2.0)

Suggest a CMA-ES/DE population size based on the number of decision variables.
Ensures the returned population is even and clamped within [`min_pop`, `max_pop`].
"""
function infer_population_size(decision_dim::Int;
                               min_pop::Int=_DEFAULT_POP_MIN,
                               max_pop::Int=_DEFAULT_POP_MAX,
                               scale::Float64=_DEFAULT_POP_SCALE)
    n = max(decision_dim, 1)
    lo = max(min_pop, 1)
    hi = max(lo, max_pop)
    raw = round(Int, scale * (4 + 3 * log(n)))
    pop = clamp(raw, lo, hi)
    iseven(pop) || (pop += 1)
    return pop
end


"""
    ensure_population_size!(os, decision_dim; kwargs...)

Ensure that `os.optimizer_settings` has a valid population, inferring one when
necessary. Returns the resolved population size (or 0 if not applicable).
"""
function ensure_population_size!(os::OptimizationSettings,
                                 decision_dim::Int;
                                 min_pop::Int=_DEFAULT_POP_MIN,
                                 max_pop::Int=_DEFAULT_POP_MAX,
                                 scale::Float64=_DEFAULT_POP_SCALE)
    oz = os.optimizer_settings
    if oz.population_size <= 0
        oz.population_size = infer_population_size(decision_dim;
                                    min_pop=min_pop,
                                    max_pop=max_pop,
                                    scale=scale)
    end
    return oz.population_size
end

function infer_sigma0(num_decision_vars::Int, scales::Dict{Symbol, Float64}; multiplier::Float64=1.0, min_sigma::Float64=0.2, max_sigma::Float64=5.0)::Float64
    num_decision_vars = max(num_decision_vars, 1)
    sigma_dim = max(sqrt(num_decision_vars) / 10, min_sigma)
    sigma_scale = max(0.1 * max_scale_value(scales), min_sigma)
    base = sqrt(sigma_dim * sigma_scale)
    return clamp(multiplier * base, min_sigma, max_sigma)
end

function ensure_sigma0!(os::OptimizationSettings, decision_dim::Int)
    oz = os.optimizer_settings
    if oz.sigma0 > 0
        return oz.sigma0
    end
    sigma = infer_sigma0(decision_dim, os.reparam_type_scales)
    oz.sigma0 = sigma
    return sigma
end

"""
Get local data path based on task type.
"""
function get_local_data_path()
    return raw"C:\Users\NinaO\.julia\dev\data\eneegma_dataset"
end
