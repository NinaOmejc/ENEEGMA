"""
Data structures for managing loaded timeseries and precomputed spectral data.
"""

"""
    Data

Container for loaded target timeseries and precomputed spectral properties used during optimization.

Holds the actual signal data, computed power spectral density (PSD), frequency grid,
measurement noise standard deviation, and optional frequency region metadata for 
region-weighted loss computation.

# Fields
- `channel::String`: Name of the target channel/component being fit
- `sampling_rate::Float64`: Sampling frequency in Hz
- `times::Vector{Float64}`: Time axis in seconds
- `signal::Vector{Float64}`: Raw timeseries data
- `measurement_noise_std::Float64`: Estimated measurement noise standard deviation (default: -1.0 if not estimated)
- `freqs::Vector{Float64}`: Frequency grid for PSD (Hz)
- `powers::Vector{Float64}`: Power values at each frequency
- `psd_representation::Symbol`: PSD representation type (`:log_power` or `:power`, default: `:log_power`)
- `freq_peak_metadata::Union{Nothing, NamedTuple}`: Optional metadata containing frequency region masks and weights for loss computation
  - If provided: contains `roi_mask`, `bg_mask`, `roi_weight`, `bg_weight`
  - If nothing: uniform (unweighted) loss is used
"""
Base.@kwdef struct Data
    channel::String
    sampling_rate::Float64
    times::Vector{Float64}
    signal::Vector{Float64}
    measurement_noise_std::Float64 = -1.0
    freqs::Vector{Float64}
    powers::Vector{Float64}
    psd_representation::Symbol = :log_power
    freq_peak_metadata::Union{Nothing, NamedTuple} = nothing
end
