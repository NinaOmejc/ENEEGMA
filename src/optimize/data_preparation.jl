"""
Data preparation utilities centralize timeseries loading, PSD computation,
and cached metadata used by optimization losses.
"""


Base.@kwdef struct TargetPSD
    channel::String
    sampling_rate::Float64
    freqs::Vector{Float64}
    powers::Vector{Float64}
    psd_representation::Symbol = :log_power
    times::Vector{Float64}
    signal::Vector{Float64}
    time_noise_metadata::Union{Nothing, NamedTuple} = nothing
    peak_metadata::Union{Nothing, NamedTuple} = nothing
end

function prepare_data!(settings::Settings)

    eeg_data = load_data(settings.data_settings)
    channel = settings.data_settings.target_channel
    fs = settings.data_settings.fs
    times = eeg_data[!, :time] # may be nothing

    signal = Float64.(eeg_data[!, Symbol(channel)])
    isempty(signal) && error("Target timeseries is empty.")

    resolved_times = times === nothing ? nothing : _normalize_time_axis(times)
    time_axis = _resolve_time_axis(resolved_times, length(signal), fs)

    ls = settings.optimization_settings.loss_settings
    freqs, powers = compute_preprocessed_welch_psd(signal, fs; loss_settings=ls)
    pipeline_has_log = psd_preproc_has_log(ls.psd_preproc)
    repr = pipeline_has_log ? :log_power : :power

    sigma_meta = maybe_initialize_std_measured_noise!(ls, signal, fs, settings.optimization_settings.loss)

    # Return TargetPSD with signal data and PSD
    return TargetPSD(
        channel=channel,
        sampling_rate=fs,
        freqs=freqs,
        powers=powers,
        psd_representation=repr,
        times=time_axis,
        signal=signal,
        time_noise_metadata=sigma_meta
    )
end


function _resolve_time_axis(times::Union{Nothing, AbstractVector{<:Real}}, n::Int, fs::Union{Nothing, Real})
    if times !== nothing
        length(times) == n || error("Time axis length $(length(times)) does not match signal length $(n).")
        return times isa Vector{Float64} ? times : collect(Float64, times)
    end
    fs === nothing && error("Cannot synthesize time axis without a sampling rate.")
    dt = 1 / Float64(fs)
    return collect(0:(n - 1)) .* dt
end

# TODO: Move to data loading utilities? Do not forget its hardcoded.
function _normalize_time_axis(times::AbstractVector)
    axis = Float64.(collect(times))
    isempty(axis) && return axis
    if maximum(abs, axis) > 120.0
        axis ./= 1000.0
    end
    return axis
end




## ------------------------------------------------------------------
## Measurement-noise heuristics
## ------------------------------------------------------------------

"""
sigma_init is a quick time‑domain estimate from the median absolute deviation of first 
differences (a robust white‑noise proxy based on sample‑to‑sample jitter).
"""
function estimate_sigma_init(data::AbstractVector{<:Real})
    n = length(data)
    n < 2 && return 0.0
    d = diff(float.(data))
    mad = median(abs.(d .- median(d)))
    return (mad / 0.6744897501960817) / sqrt(2)
end


function estimate_sigma_init(data::AbstractMatrix)
    cols = size(data, 2)
    cols == 0 && return 0.0
    acc = Float64[]
    for j in 1:cols
        push!(acc, estimate_sigma_init(view(data, :, j)))
    end
    return median(acc)
end

estimate_sigma_init(::Nothing) = nothing

"""
sigma_floor is a frequency‑domain estimate: it computes the high‑frequency noise “floor” in the PSD and scales 
a unit‑variance noise template to match that floor, so it’s tied to your Welch/PSD settings and fmin/fmax.
"""
function estimate_sigma_floor(data::AbstractVector, fs::Real, ls::LossSettings)
    fs <= 0 && return nothing
    n = length(data)
    n < 4 && return nothing

    # IMPORTANT: use the SAME kwargs as your loss uses
    fspan = (ls.fmin, ls.fmax)
    nperseg_val, nfft_val, overlap_val = _resolve_welch_params(n, Float64(fs), ls, nothing, _default_overlap())
    freqs, data_psd = compute_welch_pow_spectrum(Float64.(data), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs) && return nothing

    hf_start = max(ls.fmax - 10.0, ls.fmin)
    mask = freqs .>= hf_start
    any(mask) || return nothing
    S_data = median(data_psd[mask])
    (isfinite(S_data) && S_data > 0) || return nothing

    unit = _measurement_noise_template(n, ls)   # deterministic N(0,1)
    freqs2, unit_psd = compute_welch_pow_spectrum(Float64.(unit), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs2) && return nothing

    mask2 = freqs2 .>= hf_start
    any(mask2) || return nothing
    S_unit = median(unit_psd[mask2])
    (isfinite(S_unit) && S_unit > 0) || return nothing

    return sqrt(S_data / S_unit)
end


function estimate_sigma_floor(data::AbstractMatrix, fs::Real, loss_settings::LossSettings)
    cols = size(data, 2)
    cols == 0 && return nothing
    vals = Float64[]
    for j in 1:cols
        sigma = estimate_sigma_floor(view(data, :, j), fs, loss_settings)
        sigma === nothing || push!(vals, sigma)
    end
    isempty(vals) && return nothing
    return median(vals)
end

estimate_sigma_floor(::Nothing, ::Any, ::LossSettings) = nothing


function maybe_initialize_std_measured_noise!(ls::LossSettings,
                                              ts_data,
                                              fs::Union{Nothing, Real},
                                              loss_name::String)
    if !ls.auto_initialize_sigma_meas || ts_data === nothing
        ls.sigma_meas = -1. 
        return nothing
    end

    sigma_init = estimate_sigma_init(ts_data)
    sigma_floor = (fs === nothing) ? nothing : estimate_sigma_floor(ts_data, Float64(fs), ls)
    candidates = Float64[]
    (sigma_init !== nothing && isfinite(sigma_init) && sigma_init > 0) && push!(candidates, sigma_init)
    (sigma_floor !== nothing && isfinite(sigma_floor) && sigma_floor > 0) && push!(candidates, sigma_floor)
    isempty(candidates) && return nothing

    sigma_guess = minimum(candidates)
    resolved_sigma = (sigma_floor !== nothing && isfinite(sigma_floor) && sigma_floor > 0) ? sigma_floor : sigma_guess
    ls.sigma_meas = resolved_sigma
    return (sigma_init=sigma_init, sigma_floor=sigma_floor, sigma_guess=sigma_guess, sigma_applied=resolved_sigma)
end

