"""
Data preparation utilities centralize timeseries loading, PSD computation,
and cached metadata used by optimization losses.
"""

using Statistics
# Data struct is defined in src/types/data.jl

function prepare_data!(settings::Settings)
    eeg_data = load_data(settings.data_settings)
    fs = settings.data_settings.fs
    target_channel = settings.data_settings.target_channel
    node_names = settings.network_settings.node_names
    
    # Extract or synthesize time axis (shared across all nodes)
    signal_length_estimate = nrow(eeg_data)
    times = ENEEGMA._extract_or_synthesize_time_axis(eeg_data, fs, signal_length_estimate)
    
    # Build dict of node data
    node_data_dict = Dict{String, ENEEGMA.NodeData}()
    
    # Determine node->channel mapping
    if target_channel isa String
        # Single-node: map the single channel to the first (only) node
        channel_mapping = Dict(node_names[1] => target_channel)
    else
        # Multi-node: use provided dict
        channel_mapping = target_channel::Dict{String, String}
    end
    
    ls = settings.optimization_settings.loss_settings
    ds = settings.data_settings
    freq_peak_metadata_by_node = Dict{String, Any}()
    freq_grid_by_node = Dict{String, Vector{Float64}}()
    
    # Load data for each node that has a channel mapping
    for node_name in node_names
        if !haskey(channel_mapping, node_name)
            vwarn("Node '$node_name' not found in target_channel mapping, skipping data loading"; level=2)
            continue
        end
        
        channel = channel_mapping[node_name]
        signal = Float64.(eeg_data[!, Symbol(channel)])
        isempty(signal) && error("Target timeseries for node '$node_name' (channel '$channel') is empty.")
        
        # Ensure times matches signal length
        times_node = times[1:length(signal)]
        
        # Remove transient period
        transient_duration = ds.psd.transient_period_duration
        keep_idx = ENEEGMA.get_indices_after_transient_removal(times_node, transient_duration, times_node[1], fs)
        
        if !isempty(keep_idx)
            signal = signal[keep_idx]
            times_node = times_node[keep_idx]
        end
        
        # Compute PSD for this node
        freqs, powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs; loss_settings=ls, data_settings=ds)
        
        # Estimate measurement noise for this node
        measurement_noise_std = ENEEGMA.estimate_measurement_noise_std(signal, fs, ds, ls;
                                                                       node_name=node_name,
                                                                       channel=channel)
        
        # Compute frequency regions for this node
        roi_mode = get(ds.spectral_roi_by_node, node_name, ds.spectral_roi_definition_mode)
        freq_peak_metadata = if roi_mode == :copy
            source_node = get(ds.spectral_roi_copy_source_by_node, node_name, "")
            isempty(source_node) && error(
                "Node '$node_name' uses spectral ROI copy mode, but no source node is configured. " *
                "Use data_settings.spectral_roi = Dict(\"$node_name\" => \"copy:<source>\")."
            )
            haskey(freq_peak_metadata_by_node, source_node) || error(
                "Node '$node_name' wants to copy spectral ROI from '$source_node', but '$source_node' has not been computed yet. " *
                "Put source node before dependent node in network_settings.node_names, or use an explicit non-copy ROI."
            )
            source_freqs = freq_grid_by_node[source_node]
            length(source_freqs) == length(freqs) || error(
                "Node '$node_name' wants to copy spectral ROI from '$source_node', but their frequency-grid lengths differ " *
                "($(length(freqs)) vs $(length(source_freqs)))."
            )
            all(isapprox.(source_freqs, freqs; atol=1e-8, rtol=1e-6)) || error(
                "Node '$node_name' wants to copy spectral ROI from '$source_node', but their frequency grids do not match."
            )

            src_meta = freq_peak_metadata_by_node[source_node]
            (; src_meta..., copied_from = source_node)
        else
            ENEEGMA.compute_frequency_regions(freqs, powers, node_name, ds, ls)
        end
        freq_peak_metadata_by_node[node_name] = freq_peak_metadata
        freq_grid_by_node[node_name] = freqs
        
        # Determine PSD representation
        pipeline_has_log = ENEEGMA.psd_preproc_has_log(ds.psd.preproc_pipeline)
        repr = pipeline_has_log ? :log_power : :power
        
        # Create and store NodeData
        node_data_dict[node_name] = ENEEGMA.NodeData(
            channel=channel,
            signal=signal,
            freqs=freqs,
            powers=powers,
            measurement_noise_std=measurement_noise_std,
            psd_representation=repr,
            freq_peak_metadata=freq_peak_metadata
        )
    end
    
    # Verify at least one node was loaded
    if isempty(node_data_dict)
        error("No nodes with data loaded. Check target_channel and node_names.")
    end
    

    # Return Data with all node_data
    return Data(
        node_data=node_data_dict,
        sampling_rate=fs,
        times=times[1:length(node_data_dict[first(keys(node_data_dict))].signal)],
        # add removed transient duration to total duration for metadata completeness
        removed_transient_duration_sec=ds.psd.transient_period_duration
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


function range_mask(freq::Vector{Float64}, lower::Float64, upper::Float64)
    return (freq .>= lower) .& (freq .<= upper)
end

function fit_line(x::Vector{Float64}, y::Vector{Float64})
    length(x) == length(y) || error("Line fit vectors must have the same length")
    length(x) >= 2 || error("Need at least two points for line fit")
    x_mean = mean(x)
    y_mean = mean(y)
    denom = sum((x .- x_mean) .^ 2)
    denom > 0 || return (intercept = y_mean, slope = 0.0)
    slope = sum((x .- x_mean) .* (y .- y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return (intercept = intercept, slope = slope)
end

function _powers_are_logged(powers::AbstractVector{<:Real})::Bool
    return !isempty(powers) && any(<(0), powers)
end

function fit_aperiodic_background(freq::Vector{Float64},
                                  powers::Vector{Float64},
                                  fmin::Float64=4.0,
                                  fmax::Float64=45.0;
                                  powers_are_log::Bool=_powers_are_logged(powers),
                                  refit_residual_quantile::Float64=0.80)
    log_psd = powers_are_log ? powers : log10.(max.(powers, eps(Float64)))
    mask = range_mask(freq, fmin, fmax) .& (freq .> 0)
    fit_indices = findall(mask)
    length(fit_indices) >= 3 || return fill(mean(log_psd), length(log_psd))

    x = log10.(freq[fit_indices])
    y = log_psd[fit_indices]
    first_fit = fit_line(x, y)
    y_hat = first_fit.intercept .+ first_fit.slope .* x
    residual = y .- y_hat
    cutoff = quantile(residual, refit_residual_quantile)
    keep = residual .<= cutoff

    if count(identity, keep) >= 3
        second_fit = fit_line(x[keep], y[keep])
    else
        second_fit = first_fit
    end

    return second_fit.intercept .+ second_fit.slope .* log10.(max.(freq, eps(Float64)))
end

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

function moving_mean(x::AbstractVector, w::Int)
    w = max(1, w)
    w += iseven(w) ? 1 : 0
    h = div(w, 2)

    xf = collect(float.(x))
    out = similar(xf, Float64)

    for i in eachindex(xf)
        lo = max(firstindex(xf), i - h)
        hi = min(lastindex(xf), i + h)
        out[i] = mean(view(xf, lo:hi))
    end

    return out
end


function moving_quantile(x::AbstractVector, w::Int, q::Real)
    w = max(3, w)
    w += iseven(w) ? 1 : 0
    h = div(w, 2)

    xf = collect(float.(x))
    out = similar(xf, Float64)

    for i in eachindex(xf)
        lo = max(firstindex(xf), i - h)
        hi = min(lastindex(xf), i + h)
        out[i] = quantile(view(xf, lo:hi), q)
    end

    return out
end

# ---------------------------------------------------------------------
# Peak boundary finder
# ---------------------------------------------------------------------

"""
    peak_bounds(freqs, y, baseline, peak_idx; width_fraction=0.5, fmin=-Inf, fmax=Inf)

Find start/end of one peak.

Boundary level is:

    baseline[peak] + width_fraction * (y[peak] - baseline[peak])

Recommended:
- `width_fraction = 0.5`: core half-prominence width
- `width_fraction = 0.25`: broader ROI mask
"""
function peak_bounds(
    freqs::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    baseline::AbstractVector{<:Real},
    peak_idx::Int;
    width_fraction::Real = 0.25,
    fmin::Real = -Inf,
    fmax::Real = Inf,
)
    n = length(y)

    peak_excess = y[peak_idx] - baseline[peak_idx]

    if peak_excess <= 0
        return (
            left_idx = peak_idx,
            right_idx = peak_idx,
            left_freq = Float64(freqs[peak_idx]),
            right_freq = Float64(freqs[peak_idx]),
            width_hz = 0.0,
            boundary_level = Float64(baseline[peak_idx]),
        )
    end

    boundary_level = baseline[peak_idx] + width_fraction * peak_excess

    left = peak_idx
    while left > 1 && freqs[left] >= fmin
        if y[left] <= boundary_level
            break
        end
        left -= 1
    end

    right = peak_idx
    while right < n && freqs[right] <= fmax
        if y[right] <= boundary_level
            break
        end
        right += 1
    end

    return (
        left_idx = left,
        right_idx = right,
        left_freq = Float64(freqs[left]),
        right_freq = Float64(freqs[right]),
        width_hz = Float64(freqs[right] - freqs[left]),
        boundary_level = Float64(boundary_level),
    )
end


# ---------------------------------------------------------------------
# Spectral peak finder
# ---------------------------------------------------------------------

"""
    find_spectral_peaks(freqs, powers; kwargs...)

Detect spectral peaks using:
1. smoothing,
2. rolling lower-quantile baseline,
3. excess above baseline,
4. derivative/local-maximum candidates,
5. adaptive threshold independent of the largest peak,
6. peak bounds for ROI mask.

Returns a vector of NamedTuples containing:
    idx, freq, power, smooth_power, baseline, excess,
    left_idx, right_idx, left_freq, right_freq, width_hz
"""
function find_spectral_peaks(
    freqs::AbstractVector{<:Real},
    powers::AbstractVector{<:Real};
    fmin::Real = 1.0,
    fmax::Real = 48.0,
    max_peaks::Int = 5,
    sensitivity::Real = 0.5,
    smooth_bins::Int = 5,
    baseline_bins::Int = 31,
    baseline_quantile::Real = 0.20,
    width_fraction::Real = 0.25,
    min_width_hz::Real = 0.0,
    min_distance_hz::Real = 1.0,
)
    @assert length(freqs) == length(powers)

    sensitivity = clamp(sensitivity, 0.0, 1.0)

    freq_mask = (freqs .>= fmin) .& (freqs .<= fmax)
    freq_indices = findall(freq_mask)
    length(freq_indices) < 3 && return NamedTuple[]

    y = collect(float.(powers))

    # Smooth spectrum.
    ys = moving_mean(y, smooth_bins)

    # Estimate local baseline. Window should be wider than typical peak width.
    baseline = moving_quantile(ys, baseline_bins, baseline_quantile)

    # Positive peak component above local baseline.
    excess = ys .- baseline

    # Candidate peaks: local maxima of excess.
    candidates = Int[]

    for k in 2:(length(freq_indices) - 1)
        i = freq_indices[k]
        iprev = freq_indices[k - 1]
        inext = freq_indices[k + 1]

        if excess[i] > excess[iprev] &&
           excess[i] >= excess[inext] &&
           excess[i] > 0
            push!(candidates, i)
        end
    end

    isempty(candidates) && return NamedTuple[]

    positive_excess = excess[freq_indices][excess[freq_indices] .> 0]
    isempty(positive_excess) && return NamedTuple[]

    # Sensitivity mapping:
    # sensitivity = 0.0 -> strict threshold
    # sensitivity = 1.0 -> permissive threshold
    #
    # This is independent of the biggest peak.
    threshold_quantile = 0.90 - 0.40 * sensitivity
    threshold_quantile = clamp(threshold_quantile, 0.50, 0.95)
    min_excess = quantile(positive_excess, threshold_quantile)

    raw_peaks = NamedTuple[]

    for i in candidates
        if excess[i] < min_excess
            continue
        end

        b = peak_bounds(
            freqs,
            ys,
            baseline,
            i;
            width_fraction = width_fraction,
            fmin = fmin,
            fmax = fmax,
        )

        if b.width_hz < min_width_hz
            continue
        end

        push!(raw_peaks, (
            idx = i,
            freq = Float64(freqs[i]),
            power = Float64(y[i]),
            smooth_power = Float64(ys[i]),
            baseline = Float64(baseline[i]),
            excess = Float64(excess[i]),
            left_idx = b.left_idx,
            right_idx = b.right_idx,
            left_freq = b.left_freq,
            right_freq = b.right_freq,
            width_hz = b.width_hz,
            boundary_level = b.boundary_level,
        ))
    end

    isempty(raw_peaks) && return NamedTuple[]

    # Keep strongest peaks first.
    raw_peaks = sort(raw_peaks, by = p -> p.excess, rev = true)

    # Enforce minimum distance so one broad bump is not split into many peaks.
    selected = NamedTuple[]

    for p in raw_peaks
        far_enough = all(abs(p.freq - q.freq) >= min_distance_hz for q in selected)

        if far_enough
            push!(selected, p)
        end

        length(selected) >= max_peaks && break
    end

    # Return sorted by frequency.
    return sort(selected, by = p -> p.freq)
end


function detect_peaks_automatic(
    freqs::Vector{Float64},
    powers::Vector{Float64},
    sensitivity::Float64;
    fmin::Float64 = 1.0,
    fmax::Float64 = 48.0,
    remove_aperiodic_background::Bool = false,
    powers_are_log::Bool = _powers_are_logged(powers),
)
    n = length(freqs)
    roi_mask = falses(n)
    sensitivity = clamp(sensitivity, 0.0, 1.0)

    aperiodic_log_powers = nothing
    periodic_log_powers = nothing
    periodic_relative_powers = nothing
    detection_powers = powers

    if remove_aperiodic_background
        aperiodic_log_powers = fit_aperiodic_background(
            freqs,
            powers,
            fmin,
            fmax;
            powers_are_log = powers_are_log,
        )

        log_powers = powers_are_log ? powers : log10.(max.(powers, eps(Float64)))
        periodic_log_powers = log_powers .- aperiodic_log_powers
        periodic_relative_powers = 10.0 .^ periodic_log_powers

        # Detect peaks on periodic component.
        detection_powers = periodic_relative_powers
    end

    # Restrict to frequency range.
    freq_mask = (freqs .>= fmin) .& (freqs .<= fmax)
    freq_indices = findall(freq_mask)

    if isempty(freq_indices)
        return roi_mask, NamedTuple[]
    end

    # -----------------------------------------------------------------
    # New peak detection
    # -----------------------------------------------------------------

    peaks = find_spectral_peaks(
        freqs,
        detection_powers;
        fmin = fmin,
        fmax = fmax,
        sensitivity = sensitivity,
        max_peaks = 5,

        # Start values I would use for EEG/EMG spectra.
        smooth_bins = 5,
        baseline_bins = 31,
        baseline_quantile = 0.20,

        # ROI width.
        # 0.5 = core peak; 0.25 = wider peak support.
        width_fraction = 0.25,

        # Optional shape constraints.
        min_width_hz = 0.0,
        min_distance_hz = 1.0,
    )

    # Create ROI mask from peak bounds.
    for p in peaks
        roi_mask[p.left_idx:p.right_idx] .= true
    end

    return roi_mask, peaks
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

function manual_spectral_roi_for_node(data_settings::DataSettings, node_name::String)::Vector{Tuple{Float64, Float64}}
    if haskey(data_settings.spectral_roi_manual, node_name)
        return data_settings.spectral_roi_manual[node_name]
    elseif haskey(data_settings.spectral_roi_manual, "__all__")
        return data_settings.spectral_roi_manual["__all__"]
    end

    return Tuple{Float64, Float64}[]
end

function measurement_noise_bands_for_node(data_settings::DataSettings,
                                          node_name::String,
                                          channel::Union{Nothing, AbstractString}=nothing)::Vector{Tuple{Float64, Float64}}
    if haskey(data_settings.measurement_noise_bands, node_name)
        return data_settings.measurement_noise_bands[node_name]
    elseif channel !== nothing && haskey(data_settings.measurement_noise_bands, String(channel))
        return data_settings.measurement_noise_bands[String(channel)]
    elseif haskey(data_settings.measurement_noise_bands, "__all__")
        return data_settings.measurement_noise_bands["__all__"]
    end

    return Tuple{Float64, Float64}[]
end

"""
    compute_frequency_regions(freqs, powers, node_name, data_settings::DataSettings, ls::LossSettings)

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
                                   node_name::String,
                                   data_settings::DataSettings, ls::LossSettings)
    roi_mode = get(data_settings.spectral_roi_by_node, node_name, data_settings.spectral_roi_definition_mode)
    if roi_mode == :manual
        roi_regions = ENEEGMA.manual_spectral_roi_for_node(data_settings, node_name)
        roi_mask = ENEEGMA.build_mask_from_regions(freqs, roi_regions)
        auto_peak_info = nothing
    elseif roi_mode == :auto
        auto_peak_info = ENEEGMA.detect_peaks_automatic(
            freqs,
            powers,
            data_settings.spectral_roi_auto_peak_sensitivity;
            fmin=ls.fmin,
            fmax=ls.fmax,
            remove_aperiodic_background=data_settings.spectral_roi_auto_remove_aperiodic_background,
            powers_are_log=ENEEGMA.psd_preproc_has_log(data_settings.psd.preproc_pipeline)
        )
        roi_mask = auto_peak_info.roi_mask
    else
        error("compute_frequency_regions does not handle spectral ROI mode '$roi_mode' directly for node '$node_name'. Copy mode must be resolved in prepare_data!().")
    end
    
    bg_mask = .!roi_mask
    
    return (
        roi_mask=roi_mask,
        bg_mask=bg_mask,
        roi_weight=ls.roi_weight,
        bg_weight=ls.bg_weight,
        auto_peak_info=auto_peak_info,
        detection_powers=auto_peak_info === nothing ? nothing : auto_peak_info.detection_powers,
        aperiodic_log_powers=auto_peak_info === nothing ? nothing : auto_peak_info.aperiodic_log_powers,
        periodic_log_powers=auto_peak_info === nothing ? nothing : auto_peak_info.periodic_log_powers,
        periodic_relative_powers=auto_peak_info === nothing ? nothing : auto_peak_info.periodic_relative_powers
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
    rng = if data_settings !== nothing && hasproperty(data_settings, :psd) && data_settings.psd.noise_seed !== nothing
        Random.MersenneTwister(data_settings.psd.noise_seed)
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

function estimate_sigma_floor_in_bands(data::AbstractVector,
                                       fs::Real,
                                       data_settings::Union{Nothing, DataSettings},
                                       ls::LossSettings,
                                       bands::Vector{Tuple{Float64, Float64}})
    fs <= 0 && return nothing
    isempty(bands) && return nothing
    n = length(data)
    n < 4 && return nothing

    fspan = (ls.fmin, ls.fmax)
    nperseg_val, nfft_val, overlap_val = ENEEGMA._resolve_welch_params(n, Float64(fs), data_settings)
    freqs, data_psd = ENEEGMA.compute_welch_psd(Float64.(data), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs) && return nothing

    mask = ENEEGMA.build_mask_from_regions(freqs, bands)
    any(mask) || return nothing
    S_data = Statistics.median(data_psd[mask])
    (isfinite(S_data) && S_data > 0) || return nothing

    rng = if data_settings !== nothing && hasproperty(data_settings, :psd) && data_settings.psd.noise_seed !== nothing
        Random.MersenneTwister(data_settings.psd.noise_seed)
    else
        nothing
    end
    unit = ENEEGMA._measurement_noise_template(n, rng)
    freqs2, unit_psd = ENEEGMA.compute_welch_psd(Float64.(unit), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs2) && return nothing

    mask2 = ENEEGMA.build_mask_from_regions(freqs2, bands)
    any(mask2) || return nothing
    S_unit = Statistics.median(unit_psd[mask2])
    (isfinite(S_unit) && S_unit > 0) || return nothing

    return sqrt(S_data / S_unit)
end


function _is_valid_sigma(sigma::Union{Nothing, Real})
    """Check if sigma is a valid positive finite number."""
    return sigma !== nothing && isfinite(sigma) && sigma > 0
end


function estimate_measurement_noise_std(signal::AbstractVector{<:Real},
                                        fs::Union{Nothing, Real},
                                        data_settings::DataSettings,
                                        ls::LossSettings;
                                        node_name::Union{Nothing, AbstractString}=nothing,
                                        channel::Union{Nothing, AbstractString}=nothing)::Float64
    """
    Estimate per-node measurement noise sigma according to `data_settings.measurement_noise_mode`.

    Returns `-1.0` when estimation is disabled or no valid estimate can be formed.
    `:global_highfreq` preserves the legacy high-frequency heuristic, `:none` disables
    measurement-noise estimation, and `:node_specific` uses configured per-node/per-channel bands.
    For `:node_specific`, an empty band list (`[]`) disables estimation for that node/channel.
    """
    if !data_settings.estimate_measurement_noise || signal === nothing
        return -1.0
    end

    if data_settings.measurement_noise_mode === :none
        return -1.0
    elseif data_settings.measurement_noise_mode === :node_specific
        fs === nothing && return -1.0
        resolved_node = node_name === nothing ? "" : String(node_name)
        resolved_channel = channel === nothing ? nothing : String(channel)
        bands = ENEEGMA.measurement_noise_bands_for_node(data_settings, resolved_node, resolved_channel)
        isempty(bands) && return -1.0
        sigma_bands = estimate_sigma_floor_in_bands(signal, Float64(fs), data_settings, ls, bands)
        return _is_valid_sigma(sigma_bands) ? Float64(sigma_bands) : -1.0
    end

    sigma_init = estimate_sigma_init(signal)
    sigma_floor = fs === nothing ? nothing : estimate_sigma_floor(signal, Float64(fs), data_settings, ls)

    candidates = Float64[]
    _is_valid_sigma(sigma_init) && push!(candidates, sigma_init)
    _is_valid_sigma(sigma_floor) && push!(candidates, sigma_floor)
    isempty(candidates) && return -1.0

    sigma_guess = minimum(candidates)
    return _is_valid_sigma(sigma_floor) ? sigma_floor : sigma_guess
end

