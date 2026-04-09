"""
Data preparation utilities centralize timeseries loading, PSD computation,
and cached metadata used by optimization losses.
"""

# Data struct is defined in src/types/data.jl

function prepare_data!(settings::Settings)

    eeg_data = load_data(settings.data_settings)
    fs = settings.data_settings.fs

    # extract signal
    channel = settings.data_settings.target_channel
    signal = Float64.(eeg_data[!, Symbol(channel)])
    isempty(signal) && error("Target timeseries is empty.")

    # Extract or synthesize time axis
    times = _extract_or_synthesize_time_axis(eeg_data, fs, length(signal))
    
    # Extract settings early (needed for time axis resolution)
    ls = settings.optimization_settings.loss_settings
    freqs, powers = compute_preprocessed_welch_psd(signal, fs; loss_settings=ls)
    pipeline_has_log = psd_preproc_has_log(ls.psd_preproc)
    repr = pipeline_has_log ? :log_power : :power

    maybe_initialize_std_measured_noise!(settings.data_settings, ls, signal, fs)

    # Compute frequency regions for weighted loss
    freq_peak_metadata = compute_frequency_regions(freqs, powers, settings.data_settings, ls)

    # Return Data with signal, PSD, measurement noise, and region masks
    return Data(
        channel=channel,
        times=times,
        signal=signal,
        sampling_rate=fs,
        freqs=freqs,
        powers=powers,
        psd_representation=repr,
        measurement_noise_std=ls.measurement_noise_std,
        freq_peak_metadata=freq_peak_metadata
    )
end


"""
    _extract_or_synthesize_time_axis(data, fs, signal_length)

Extract time column from data, or synthesize from sampling rate if unavailable.

Tries to find a time column with names: 'time', 't', or 'times'.
If found, returns it as Float64 vector.
If not found, synthesizes time axis from sampling rate (fs).
Raises error if neither available.

# Returns
- `Vector{Float64}`: Time axis in seconds
"""
function _extract_or_synthesize_time_axis(data, fs, signal_length::Int)
    for col_name in [:time, :t, :times]
        if haspropnn(data, col_name)
            return Float64.(collect(data[!, col_name]))
        end
    end
    
    fs === nothing && error("No time column found in data, and no sampling rate (fs) provided in settings. Please add a time column ('time', 't', or 'times') or set fs in settings.")
    vwarn("No time column found. Synthesizing time axis from sampling rate (fs).")
    return _synthesize_time_axis(signal_length, fs)
end

"""
    _synthesize_time_axis(n, fs)

Synthesize uniform time axis from sampling rate: [0, dt, 2dt, ..., (n-1)dt] where dt = 1/fs.

# Arguments
- `n::Int`: Number of samples
- `fs::Real`: Sampling rate in Hz

# Returns
- `Vector{Float64}`: Time axis in seconds
"""
function _synthesize_time_axis(n::Int, fs::Real)
    dt = 1 / Float64(fs)
    return collect(0:(n - 1)) .* dt
end


## ------------------------------------------------------------------
## Frequency Region Definition (ROI vs Background)
## ------------------------------------------------------------------

"""
    detect_peaks_automatic(freqs, powers, sensitivity; fmin=1.0, fmax=48.0)

Automatically detect frequency peaks in the PSD for ROI masking.

# Arguments
- `freqs::Vector{Float64}`: Frequency axis
- `powers::Vector{Float64}`: Power spectrum
- `sensitivity::Float64`: Detection sensitivity (0.0-1.0, higher=looser)
- `fmin`, `fmax`: Frequency range to search

# Returns
- `BitVector`: Mask where peaks are marked as `true` (ROI)
"""
function detect_peaks_automatic(freqs::Vector{Float64}, powers::Vector{Float64}, sensitivity::Float64;
                               fmin::Float64=1.0, fmax::Float64=48.0)
    n = length(freqs)
    roi_mask = falses(n)
    sensitivity = clamp(sensitivity, 0.0, 1.0)
    
    # Restrict to frequency range
    freq_mask = (fmin .<= freqs .<= fmax)
    freq_indices = findall(freq_mask)
    isempty(freq_indices) && return roi_mask
    
    powers_in_range = powers[freq_indices]
    
    # Compute threshold: baseline + (peak - baseline) * (1 - sensitivity)
    baseline = quantile(powers_in_range, 0.2)  # lower quartile as baseline
    peak = maximum(powers_in_range)
    threshold = baseline + (peak - baseline) * (1.0 - sensitivity)
    
    # Mark peaks above threshold
    for (local_idx, global_idx) in enumerate(freq_indices)
        if powers[global_idx] >= threshold
            roi_mask[global_idx] = true
        end
    end
    
    return roi_mask
end

"""
    build_mask_from_regions(freqs, regions)

Build frequency mask from manually specified regions.

# Arguments
- `freqs::Vector{Float64}`: Frequency axis
- `regions::Vector{Tuple{Float64, Float64}}`: List of (fmin, fmax) tuples

# Returns
- `BitVector`: Mask where manual regions are marked as `true` (ROI)
"""
function build_mask_from_regions(freqs::Vector{Float64}, regions::Vector{Tuple{Float64, Float64}})
    roi_mask = falses(length(freqs))
    for (fmin, fmax) in regions
        for (i, f) in enumerate(freqs)
            if fmin <= f <= fmax
                roi_mask[i] = true
            end
        end
    end
    return roi_mask
end

"""
    compute_frequency_regions(freqs, powers, data_settings::DataSettings, ls::LossSettings)

Compute ROI and background masks for weighted loss based on data settings.

# Arguments
- `freqs::Vector{Float64}`: Frequency axis
- `powers::Vector{Float64}`: Power spectrum
- `data_settings::DataSettings`: Contains region_definition_mode and related config
- `ls::LossSettings`: Contains fmin, fmax for frequency range

# Returns
- `NamedTuple` with:
  - `roi_mask::BitVector`: Regions of interest
  - `bg_mask::BitVector`: Background regions
  - `roi_weight::Float64`: Weight for ROI loss
  - `bg_weight::Float64`: Weight for background loss
"""
function compute_frequency_regions(freqs::Vector{Float64}, powers::Vector{Float64},
                                   data_settings::DataSettings, ls::LossSettings)
    if data_settings.region_definition_mode == :manual
        roi_mask = build_mask_from_regions(freqs, data_settings.manual_frequency_regions)
    else  # :auto
        roi_mask = detect_peaks_automatic(freqs, powers, data_settings.auto_peak_sensitivity;
                                         fmin=ls.fmin, fmax=ls.fmax)
    end
    
    bg_mask = .!roi_mask
    
    return (
        roi_mask=roi_mask,
        bg_mask=bg_mask,
        roi_weight=ls.roi_weight,
        bg_weight=ls.bg_weight
    )
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


function _is_valid_sigma(sigma::Union{Nothing, Real})
    """Check if sigma is a valid positive finite number."""
    return sigma !== nothing && isfinite(sigma) && sigma > 0
end


function maybe_initialize_std_measured_noise!(data_settings::DataSettings,
                                              ls::LossSettings,
                                              ts_data,
                                              fs::Union{Nothing, Real})
    """
    Estimate measurement noise sigma from data using time and frequency-domain methods.
    
    If disabled or data unavailable, sets measurement_noise_std to -1.0.
    Otherwise, combines time-domain (MAD-based) and frequency-domain (PSD floor) estimates,
    preferring the frequency-domain estimate if available.
    """
    if !data_settings.estimate_measurement_noise || ts_data === nothing
        ls.measurement_noise_std = -1.0
        return nothing
    end

    # Estimate sigma from two independent methods
    sigma_init = estimate_sigma_init(ts_data)
    sigma_floor = (fs === nothing) ? nothing : estimate_sigma_floor(ts_data, Float64(fs), ls)
    
    # Collect valid estimates
    candidates = Float64[]
    _is_valid_sigma(sigma_init) && push!(candidates, sigma_init)
    _is_valid_sigma(sigma_floor) && push!(candidates, sigma_floor)
    isempty(candidates) && return nothing

    # Use frequency-domain estimate if available (more robust), otherwise use minimum
    sigma_guess = minimum(candidates)
    resolved_sigma = _is_valid_sigma(sigma_floor) ? sigma_floor : sigma_guess
    
    ls.measurement_noise_std = resolved_sigma
    return nothing
end

