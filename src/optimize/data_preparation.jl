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
    times = ENEEGMA._extract_or_synthesize_time_axis(eeg_data, fs, length(signal))
    
    # Extract settings early (needed for time axis resolution)
    ls = settings.optimization_settings.loss_settings
    loss_settings = settings.optimization_settings.loss_settings
    ds = settings.data_settings
    freqs, powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs; loss_settings=ls, data_settings=ds)
    pipeline_has_log = ENEEGMA.psd_preproc_has_log(ds.psd.preproc_pipeline)
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
        measurement_noise_std=ds.measurement_noise_std,
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

Validates that the time column is likely in seconds by comparing the implied 
sampling rate (from time column) with the provided sampling rate (fs).

# Returns
- `Vector{Float64}`: Time axis in seconds
"""
function _extract_or_synthesize_time_axis(data, fs, signal_length::Int)
    for col_name in [:time, :t, :times]
        if haspropnn(data, col_name)
            time_col = Float64.(collect(data[!, col_name]))
            _check_time_axis_units(time_col, fs, signal_length)
            return time_col
        end
    end
    
    fs === nothing && error("No time column found in data, and no sampling rate (fs) provided in settings. Please add a time column ('time', 't', or 'times') or set fs in settings.")
    vwarn("No time column found. Synthesizing time axis from sampling rate (fs).")
    return _synthesize_time_axis(signal_length, fs)
end

"""
    _check_time_axis_units(time_col, fs, signal_length)

Validate that time column is likely in seconds by comparing implied vs provided sampling rates.

Computes the sampling rate implied by the time column and compares it with the provided fs.
Issues a warning if they differ significantly, suggesting the time column may be in 
different units (milliseconds, minutes, etc.).

# Arguments
- `time_col::Vector{Float64}`: Extracted time column
- `fs::Real`: Provided sampling rate in Hz
- `signal_length::Int`: Length of signal (used as fallback check)
"""
function _check_time_axis_units(time_col::Vector{Float64}, fs::Union{Nothing, Real}, signal_length::Int)
    fs === nothing && return
    fs <= 0 && return
    length(time_col) < 2 && return
    
    actual_duration = time_col[end] - time_col[1]
    actual_duration <= 0 && return
    
    # Compute implied sampling rate from time column
    implied_fs = (length(time_col) - 1) / actual_duration
    ratio = implied_fs / fs
    
    # Warn if ratio suggests unit mismatch
    if ratio > 900 && ratio < 1100
        vwarn("Time column may be in milliseconds instead of seconds (implied sampling rate ≈ $(round(Int, implied_fs)) Hz vs expected $(round(Int, fs)) Hz). Please verify your time axis.", level=1)
    elseif ratio > 50
        vwarn("Time column may be in different units than seconds (implied sampling rate ≈ $(round(Int, implied_fs)) Hz vs expected $(round(Int, fs)) Hz). Please verify your time axis.", level=1)
    elseif ratio < 0.01
        vwarn("Time column may be in minutes or a larger unit instead of seconds (implied sampling rate ≈ $(round(Int, implied_fs)) Hz vs expected $(round(Int, fs)) Hz). Please verify your time axis.", level=1)
    end
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
- `data_settings::DataSettings`: Contains spectral_roi_definition_mode and related config
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
    if data_settings.spectral_roi_definition_mode == :manual
        roi_mask = build_mask_from_regions(freqs, data_settings.spectral_roi_manual)
    else  # :auto
        roi_mask = detect_peaks_automatic(freqs, powers, data_settings.spectral_roi_auto_peak_sensitivity;
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
function estimate_sigma_floor(data::AbstractVector, fs::Real, data_settings::Union{Nothing, DataSettings}, ls::LossSettings)
    fs <= 0 && return nothing
    n = length(data)
    n < 4 && return nothing

    # IMPORTANT: use the SAME kwargs as your loss uses
    fspan = (ls.fmin, ls.fmax)
    nperseg_val, nfft_val, overlap_val = ENEEGMA._resolve_welch_params(n, Float64(fs), data_settings)
    freqs, data_psd = ENEEGMA.compute_welch_psd(Float64.(data), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs) && return nothing

    hf_start = max(ls.fmax - 10.0, ls.fmin)
    mask = freqs .>= hf_start
    any(mask) || return nothing
    S_data = Statistics.median(data_psd[mask])
    (isfinite(S_data) && S_data > 0) || return nothing

    # Create appropriately seeded RNG for noise template
    rng = if data_settings !== nothing && data_settings.psd_noise_seed !== nothing
        Random.MersenneTwister(data_settings.psd_noise_seed)
    else
        nothing
    end
    unit = ENEEGMA._measurement_noise_template(n, rng)   # deterministic N(0,1) if seed is set
    freqs2, unit_psd = ENEEGMA.compute_welch_psd(Float64.(unit), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs2) && return nothing

    mask2 = freqs2 .>= hf_start
    any(mask2) || return nothing
    S_unit = Statistics.median(unit_psd[mask2])
    (isfinite(S_unit) && S_unit > 0) || return nothing

    return sqrt(S_data / S_unit)
end


function estimate_sigma_floor(data::AbstractMatrix, fs::Real, data_settings::Union{Nothing, DataSettings}, loss_settings::LossSettings)
    cols = size(data, 2)
    cols == 0 && return nothing
    vals = Float64[]
    for j in 1:cols
        sigma = estimate_sigma_floor(view(data, :, j), fs, data_settings, loss_settings)
        sigma === nothing || push!(vals, sigma)
    end
    isempty(vals) && return nothing
    return median(vals)
end

estimate_sigma_floor(::Nothing, ::Any, ::Union{Nothing, DataSettings}, ::LossSettings) = nothing


function _is_valid_sigma(sigma::Union{Nothing, Real})
    """Check if sigma is a valid positive finite number."""
    return sigma !== nothing && isfinite(sigma) && sigma > 0
end


function maybe_initialize_std_measured_noise!(data_settings::DataSettings,
                                              ls::LossSettings,
                                              signal::AbstractVector{<:Real},
                                              fs::Union{Nothing, Real})
    """
    Estimate measurement noise sigma from data using time and frequency-domain methods.
    
    If disabled or data unavailable, sets measurement_noise_std to -1.0.
    Otherwise, combines time-domain (MAD-based) and frequency-domain (PSD floor) estimates,
    preferring the frequency-domain estimate if available.
    """
    if !data_settings.estimate_measurement_noise || signal === nothing
        data_settings.measurement_noise_std = -1.0
        return nothing
    end

    # Estimate sigma from two independent methods
    sigma_init = estimate_sigma_init(signal)
    sigma_floor = (fs === nothing) ? nothing : estimate_sigma_floor(signal, Float64(fs), data_settings, ls)

    # Collect valid estimates
    candidates = Float64[]
    _is_valid_sigma(sigma_init) && push!(candidates, sigma_init)
    _is_valid_sigma(sigma_floor) && push!(candidates, sigma_floor)
    isempty(candidates) && return nothing

    # Use frequency-domain estimate if available (more robust), otherwise use minimum
    sigma_guess = minimum(candidates)
    resolved_sigma = _is_valid_sigma(sigma_floor) ? sigma_floor : sigma_guess
    
    data_settings.measurement_noise_std = resolved_sigma
    return nothing
end

