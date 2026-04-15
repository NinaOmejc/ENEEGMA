# Global verbose flag and helpers
const verbose = Ref(false)
const verbosity_level = Ref(0)
const _task_settings = Dict{Task, Any}()

function set_verbose(flag::Bool)
    verbose[] = flag
    verbosity_level[] = flag ? max(verbosity_level[], 1) : 0
end

function set_verbose(level::Int)
    clamped = clamp(level, 0, 2)
    verbosity_level[] = clamped
    verbose[] = clamped > 0
end

function set_task_settings(settings::Any)
    _task_settings[current_task()] = settings
    if hasfield(typeof(settings), :general_settings)
        gs = settings.general_settings
        if hasproperty(gs, :verbosity_level)
            set_verbose(gs.verbosity_level)
        end
    end
end

"""
    is_verbose(level)::Bool

Check verbosity status. Uses settings.general_settings.verbosity_level if available,
otherwise falls back to the global verbosity_level flag.
"""
function _task_verbosity_level()
    task = current_task()
    if haskey(_task_settings, task)
        settings = _task_settings[task]
        if hasfield(typeof(settings), :general_settings)
            gs = settings.general_settings
            if hasproperty(gs, :verbosity_level)
                return gs.verbosity_level
            end
        end
    end
    return verbosity_level[]
end

function is_verbose(level::Int=1)::Bool
    return _task_verbosity_level() >= level
end

function _as_string(msg)
    msg isa AbstractString && return msg
    return string(msg)
end

function vwarn(msg; level::Int=1)
    is_verbose(level) || return
    prefix = level == 1 ? "⚠ WARNING" : "  ⚠"
    println(prefix * ": " * _as_string(msg))
end

function vinfo(msg; level::Int=1)
    is_verbose(level) || return
    if level == 1
        # Level 1: High visibility for major milestones
        println("=== " * _as_string(msg) * " ===")
    else
        # Level 2: Compact format for details
        println("--- " * _as_string(msg) * " ---")
    end
end

function verror(msg; level::Int=1)
    is_verbose(level) || return
    prefix = level == 1 ? "❌ ERROR" : "  ❌"
    println(prefix * ": " * _as_string(msg))
end

# make_rng: create a RNG from maybe_seed
function make_rng(maybe_seed::Union{Nothing,Int,Symbol})
    if isnothing(maybe_seed)
        return Random.default_rng()       # uses current global state
    elseif maybe_seed === :auto
        # generate + log a fresh seed
        seed = rand(RandomDevice(), UInt32)
        vinfo("Auto-generated seed: $(seed)")
        return MersenneTwister(seed)
    else
        return MersenneTwister(maybe_seed)  # deterministic
    end
end



# Helper to center text (reuse from build_node.jl if not already imported)
function center(text::String, width::Int)
    pad = max(0, width - length(text))
    lpad = div(pad, 2)
    rpad = pad - lpad
    return repeat(" ", lpad) * text * repeat(" ", rpad)
end



function get_freq_bands()::Dict{String, Tuple{Float64, Float64}}
    return Dict(
            "ultralow" => (0.0, 1.0),
            "delta" => (1., 4.0),
            "theta" => (4.0, 8.0), 
            "alpha" => (8.0, 13.0),
            "betalow" => (13.0, 20.0),
            "betahigh" => (20.0, 30.0),
            "gammalow" => (30.0, 48.0),
            "gammahigh" => (52.0, 70.0),
            "ultrahigh" => (70.0, 150.0)
            )
end

"""
    haspropnn(obj, prop::Symbol)::Bool

Check if an object has a property and if that property is not `nothing`.

# Arguments
- `obj`: Object to check
- `prop::Symbol`: Property name as a symbol

# Returns
- `Bool`: `true` if property exists and is not `nothing`, `false` otherwise
"""
function haspropnn(obj, prop::Symbol)::Bool
    return hasproperty(obj, prop) && getproperty(obj, prop) !== nothing
end


"""
    get_eeg_signal(settings::Settings, df::DataFrame)::String

Get the signal name to plot from settings.

Returns the EEG output specification if defined in settings, otherwise returns
the name of the first state variable (second column after :time).

# Arguments
- `settings::Settings`: Settings object containing eeg_output specification
- `df::DataFrame`: DataFrame with time series data (columns: time + state variables)

# Returns
- `String`: Signal name to use for plotting

# Example
```julia
signal = get_eeg_signal(settings, df)
plot(df.time, df[!, Symbol(signal)])
```
"""
function get_eeg_signal(settings::Settings, df::DataFrame)::String
    if isempty(settings.network_settings.eeg_output)
        return names(df)[2]  # Default: First state variable (second column after :time)
    else
        return settings.network_settings.eeg_output
    end
end

