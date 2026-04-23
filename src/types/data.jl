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
