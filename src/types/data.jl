"""
Data structures for managing loaded timeseries and precomputed spectral data.
"""

"""
    NodeData

Per-node container for signal and spectral properties.

# Fields
- `channel::String`: Name of the data channel this node's signal came from
- `signal::Vector{Float64}`: Raw timeseries data for this node
- `freqs::Vector{Float64}`: Frequency grid for PSD (Hz)
- `powers::Vector{Float64}`: Power values at each frequency
- `measurement_noise_std::Float64`: Estimated measurement noise standard deviation
- `psd_representation::Symbol`: PSD representation type (`:log_power` or `:power`)
- `freq_peak_metadata::Union{Nothing, NamedTuple}`: Optional frequency region masks and weights
  - If provided: contains `roi_mask`, `bg_mask`, `roi_weight`, `bg_weight`
  - If nothing: uniform (unweighted) loss is used
"""
Base.@kwdef struct NodeData
    channel::String
    signal::Vector{Float64}
    freqs::Vector{Float64}
    powers::Vector{Float64}
    measurement_noise_std::Float64 = -1.0
    psd_representation::Symbol = :log_power
    freq_peak_metadata::Union{Nothing, NamedTuple} = nothing
end

"""
    Data

Container for all nodes' data and shared metadata.

Holds the signal data for each node, precomputed PSDs, and timing information
shared across all nodes. Always uses a dict-based structure where keys are node names
and values are NodeData objects containing per-node information.

# Fields
- `node_data::Dict{String, NodeData}`: Mapping of node names to their respective NodeData
  - Single-node: Dict with one entry (e.g., {"IC3" => NodeData(...)})
  - Multi-node: Dict with multiple entries (e.g., {"C" => NodeData(...), "M" => NodeData(...)})
- `sampling_rate::Float64`: Sampling frequency in Hz (shared across all nodes)
- `times::Vector{Float64}`: Time axis in seconds (shared across all nodes)
"""
Base.@kwdef struct Data
    node_data::Dict{String, NodeData}
    sampling_rate::Float64
    times::Vector{Float64}
    removed_transient_duration_sec::Float64 = 0.0
end

function _mask_to_frequency_regions(freqs::AbstractVector{<:Real},
                                    mask::AbstractVector{Bool})::Vector{Tuple{Float64, Float64}}
    length(freqs) == length(mask) || return Tuple{Float64, Float64}[]
    isempty(freqs) && return Tuple{Float64, Float64}[]

    regions = Tuple{Float64, Float64}[]
    in_region = false
    region_start = 0.0

    for i in eachindex(freqs, mask)
        if mask[i] && !in_region
            region_start = Float64(freqs[i])
            in_region = true
        elseif !mask[i] && in_region
            push!(regions, (region_start, Float64(freqs[i - 1])))
            in_region = false
        end
    end

    if in_region
        push!(regions, (region_start, Float64(freqs[end])))
    end

    return regions
end

function _format_frequency_regions(regions::Vector{Tuple{Float64, Float64}})::String
    isempty(regions) && return "[]"
    formatted = ["[$(round(fmin; digits=3)), $(round(fmax; digits=3))]" for (fmin, fmax) in regions]
    return "[" * join(formatted, ", ") * "]"
end

function Base.show(io::IO, data::Data)
    node_names = sort!(collect(keys(data.node_data)))
    n_nodes = length(node_names)
    n_samples = length(data.times)
    time_start = n_samples > 0 ? first(data.times) : NaN
    time_end = n_samples > 0 ? last(data.times) : NaN

    println(io, "\n" * "="^80)
    println(io, "Data: $(n_nodes) node$(n_nodes == 1 ? "" : "s")")
    println(io, "="^80)
    println(io, "Sampling Rate: $(data.sampling_rate) Hz")
    println(io, "Samples: $n_samples")
    println(io, "Time Range: [$time_start, $time_end] s")
    println(io, "Removed Transient: $(data.removed_transient_duration_sec) s")

    if n_nodes == 0
        println(io, "\n[Node Data]")
        println(io, "  <empty>")
        return nothing
    end

    println(io, "\n[Node Data]")
    for node_name in node_names
        node_info = data.node_data[node_name]
        signal_len = length(node_info.signal)
        freqs_len = length(node_info.freqs)
        powers_len = length(node_info.powers)
        freq_start = freqs_len > 0 ? first(node_info.freqs) : NaN
        freq_end = freqs_len > 0 ? last(node_info.freqs) : NaN
        noise_str = isfinite(node_info.measurement_noise_std) && node_info.measurement_noise_std >= 0 ?
            string(node_info.measurement_noise_std) : "not estimated"

        println(io, "  - $node_name")
        println(io, "      channel: $(node_info.channel)")
        println(io, "      signal samples: $signal_len")
        println(io, "      PSD bins: $freqs_len")
        println(io, "      PSD powers: $powers_len")
        println(io, "      frequency range: [$freq_start, $freq_end] Hz")
        println(io, "      representation: $(node_info.psd_representation)")
        println(io, "      measurement noise std: $noise_str")
        if node_info.freq_peak_metadata === nothing
            println(io, "      peak metadata: none")
        else
            pm = node_info.freq_peak_metadata
            roi_mask = Bool.(vec(pm.roi_mask))
            bg_mask = Bool.(vec(pm.bg_mask))
            roi_regions = _mask_to_frequency_regions(node_info.freqs, roi_mask)
            bg_regions = _mask_to_frequency_regions(node_info.freqs, bg_mask)
            println(io, "      peak metadata:")
            println(io, "        roi regions: $(_format_frequency_regions(roi_regions))")
            println(io, "        bg regions: $(_format_frequency_regions(bg_regions))")
            println(io, "        roi bins: $(count(roi_mask))")
            println(io, "        bg bins: $(count(bg_mask))")
            println(io, "        roi weight: $(pm.roi_weight)")
            println(io, "        bg weight: $(pm.bg_weight)")
        end
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", data::Data)
    Base.show(io, data)
end
