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

# ============================================================================
# DISPLAY FORMATTING UTILITIES
# ============================================================================

"""
    round_for_display(value::Real; sig_figs::Int=3)::Float64

Round a number to specified significant figures for cleaner display.

Useful for printing parameter defaults, bounds, and initial condition ranges
to avoid cluttered output from floating-point precision artifacts.

# Arguments
- `value::Real`: Number to round
- `sig_figs::Int`: Number of significant figures (default: 3)

# Returns
Rounded value as Float64

# Examples
```julia
round_for_display(0.123456)      # → 0.123
round_for_display(1.23456)       # → 1.23
round_for_display(0.001234)      # → 0.00123
round_for_display(12345.6)       # → 12300.0
round_for_display(0.0)           # → 0.0
```
"""
function round_for_display(value::Real; sig_figs::Int=3)::Float64
    value == 0 && return 0.0
    # Calculate the exponent (power of 10)
    exponent = floor(Int, log10(abs(value)))
    # Round to sig_figs - 1 decimal places relative to the exponent
    magnitude = 10.0 ^ (exponent - sig_figs + 1)
    return round(Float64(value) / magnitude) * magnitude
end

"""
    _task_verbosity_level()::Int

Get the current task's verbosity level from task settings, or fall back to global verbosity.
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
    get_eeg_output_indices(net::Network, settings::Settings)::Dict{String, Int}

Map node names to the state variable indices for their EEG outputs.

For each node in the network, returns the index of the state variable specified
by eeg_output (if set) or the node's brain_source field (if not set).

# Arguments
- `net::Network`: The network structure
- `settings::Settings`: Settings containing optional eeg_output dictionary

# Returns
- `Dict{String, Int}`: Mapping of node names to state variable indices

# Example
```julia
indices = get_eeg_output_indices(net, settings)
# Returns: Dict("N1" => 3, "N2" => 8)  (indices in the state vector)
```
"""
function get_eeg_output_indices(net::Network, settings::Settings)::Dict{String, Int}
    indices = Dict{String, Int}()
    state_syms = get_symbols(get_state_vars(net.vars); sort=true)
    state_syms_sym = Symbol.(state_syms)
    
    for (idx, node) in enumerate(net.nodes)
        node_name = String(node.name)
        
        # Determine expression: eeg_output (if set) or brain_source (fallback)
        output_expr = get(settings.network_settings.eeg_output, node_name, "")
        if isempty(output_expr)
            output_expr = String(node.brain_source)
        end
        
        if !isempty(output_expr)
            # For simple expressions (single variable name), find its index
            output_symbol = Symbol(strip(output_expr))
            state_idx = findfirst(==(output_symbol), state_syms_sym)
            
            if state_idx !== nothing && state_idx >= 1
                indices[node_name] = state_idx
            else
                # Fallback: use node index if we can't find the specific state
                indices[node_name] = idx
            end
        else
            # If no output specified, use first state of this node as fallback
            indices[node_name] = idx
        end
    end
    
    return indices
end

# ============================================================================
# TRANSIENT PERIOD UTILITIES
# ============================================================================

"""
    get_indices_after_transient_removal(sol_t::Vector, transient_duration::Float64, tspan_start::Float64, fs::Union{Nothing, Float64}=nothing)::AbstractVector{Int}

Compute indices to keep after removing transient period. Optimized for speed (called millions of times).

Uses sample-based counting: transient_samples = round(transient_duration * fs) then returns (transient_samples+1):end.

# Arguments
- `fs::Union{Nothing, Float64}`: Sampling frequency in Hz. If nothing, infers from time vector.
- Other args kept for API compatibility but currently unused for efficiency.
"""
function get_indices_after_transient_removal(sol_t::Vector, transient_duration::Float64, tspan_start::Float64, fs::Union{Nothing, Float64}=nothing)::AbstractVector{Int}
    if transient_duration <= 0.0
        return 1:length(sol_t)
    end
    
    # Use provided fs, otherwise infer from time vector mean spacing
    fs_use = if fs !== nothing
        fs
    else
        length(sol_t) < 2 ? 1.0 : 1.0 / ((sol_t[end] - sol_t[1]) / (length(sol_t) - 1))
    end
    
    # Sample-based counting: deterministic, no time-comparison ambiguity
    transient_samples = Int(round(transient_duration * fs_use))
    keep_start = min(transient_samples + 1, length(sol_t))
    
    return keep_start:length(sol_t)
end






