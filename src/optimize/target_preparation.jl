"""
Target preparation utilities centralize timeseries loading, PSD computation,
and cached metadata used by optimization losses.
"""

"""
    apply_subject_specific_peak_range!(settings::Settings)

Load metadata file and update peak frequency range based on subject-specific values.
If metadata_file is specified in data_settings, loads the metadata CSV, finds the
row matching the current subject/task/IC, and updates loss_settings with subject-specific
peak_min_freq_hz and peak_max_freq_hz.
"""
function apply_subject_specific_peak_range!(settings::Settings)
    ds = settings.data_settings
    ls = settings.optimization_settings.loss_settings
    
    # Check if metadata file is specified
    if ds.metadata_file === nothing
        return  # No metadata file, skip
    end
    
    # Construct full metadata file path
    metadata_path = ds.metadata_path === nothing ? ds.data_path : ds.metadata_path
    metadata_file = joinpath(metadata_path, ds.metadata_file)
    
    if !isfile(metadata_file)
        vprint("Warning: Metadata file not found: $(metadata_file). Skipping peak range update.", level=2)
        return
    end
    
    # Load metadata
    metadata = CSV.read(metadata_file, DataFrame)
    
    # Extract subject info from data_fname (e.g., "rest_sub-1_IC3.csv" -> "sub-1")
    sub_match = match(r"sub-(\d+)", ds.data_fname)
    if sub_match === nothing
        vprint("Warning: Could not extract subject ID from data_fname: $(ds.data_fname). Skipping peak range update.", level=2)
        return
    end
    sub_id = parse(Int, sub_match.captures[1])
    
    # Extract IC from target_channel (e.g., "IC3" -> 3)
    ic_match = match(r"IC(\d+)", ds.target_channel)
    if ic_match === nothing
        vprint("Warning: Could not extract IC from target_channel: $(ds.target_channel). Skipping peak range update.", level=2)
        return
    end
    ic_id = parse(Int, ic_match.captures[1])
    
    # Filter metadata for matching row
    matching_rows = filter(row -> row.sub == sub_id && row.task == ds.task_type && row.IC == ic_id, metadata)
    
    if nrow(matching_rows) == 0
        vprint("Warning: No metadata found for sub=$(sub_id), task=$(ds.task_type), IC=$(ic_id). Skipping peak range update.", level=2)
        return
    elseif nrow(matching_rows) > 1
        vprint("Warning: Multiple metadata rows found for sub=$(sub_id), task=$(ds.task_type), IC=$(ic_id). Using first match.", level=2)
    end
    
    # Get the first matching row
    metadata_row = matching_rows[1, :]
    
    # Update loss settings with subject-specific peak range
    if haskey(metadata_row, :peak_min_freq_hz) && haskey(metadata_row, :peak_max_freq_hz)
        ls.peak_min_frequency_hz = metadata_row.peak_min_freq_hz
        ls.peak_max_frequency_hz = metadata_row.peak_max_freq_hz
        ls.peak_detection_empty = true  # Use fixed range instead of detection
        vprint("Applied subject-specific peak range: $(ls.peak_min_frequency_hz)-$(ls.peak_max_frequency_hz) Hz", level=2)
    else
        vprint("Warning: Metadata row missing peak frequency columns. Skipping peak range update.", level=2)
    end
end


Base.@kwdef struct TargetPSD
    channel::String
    sampling_rate::Float64
    freqs::Vector{Float64}
    powers::Vector{Float64}
    full_powers::Vector{Float64}
    background_power::Vector{Float64}
    peak_power::Vector{Float64}
    psd_representation::Symbol = :log_power
    times::Vector{Float64}
    signal::Vector{Float64}
    time_noise_metadata::Union{Nothing, NamedTuple} = nothing
    peak_metadata::Union{Nothing, NamedTuple} = nothing
end

function prepare_target!(settings::Settings)

    # Apply subject-specific peak range from metadata if available
    apply_subject_specific_peak_range!(settings)

    eeg_data = load_data(settings.data_settings)
    channel = settings.data_settings.target_channel
    fs = settings.data_settings.fs
    times = eeg_data[!, :time] # may be nothing

    signal = Float64.(eeg_data[!, Symbol(channel)])
    isempty(signal) && error("Target timeseries is empty.")

    resolved_times = times === nothing ? nothing : ENMEEG._normalize_time_axis(times)
    time_axis = ENMEEG._resolve_time_axis(resolved_times, length(signal), fs)

    ls = settings.optimization_settings.loss_settings
    freqs, powers = compute_preprocessed_welch_psd(signal, fs; loss_settings=ls)
    pipeline_has_log = ENMEEG.psd_preproc_has_log(ls.psd_preproc)
    repr = pipeline_has_log ? :log_power : :power

    sigma_meta = maybe_initialize_std_measured_noise!(ls, signal, fs, settings.optimization_settings.loss)

    if settings.data_settings.task_type == "rest"
        background_quantile = 0.5
    else
        background_quantile = 0.1
    end
    comps = separate_psd_components(freqs, powers, ls, background_quantile=background_quantile)

    if settings.optimization_settings.component_fit == "rest"
        chosen_powers = comps.log_peak
    elseif settings.optimization_settings.component_fit == "background"
        chosen_powers = comps.log_background
    elseif settings.optimization_settings.component_fit == "ssvep"
        chosen_powers = comps.log_peak
    elseif settings.optimization_settings.component_fit == "full" || settings.optimization_settings.component_fit == "all"
        chosen_powers = powers
    else
        error("Unknown component_fit option: $(settings.optimization_settings.component_fit)")
    end
    vprint("Prepared target PSD component $(settings.optimization_settings.component_fit) with $(length(freqs)) frequency bins.", level=2)

    peak_meta = build_broad_peak_metadata(freqs, chosen_powers, ls, settings.optimization_settings.loss)

    return TargetPSD(channel=channel,
                     sampling_rate=fs,
                     freqs=freqs,
                     powers=chosen_powers,
                     full_powers=powers,
                     background_power=comps.log_background,
                     peak_power=comps.log_peak,
                     psd_representation=repr,
                     times=time_axis,
                     signal=signal,
                     time_noise_metadata=sigma_meta,
                     peak_metadata=peak_meta)
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


# ------------------------------------------------------------------
# Peak/background helpers
# ------------------------------------------------------------------

const _PEAK_LOSS_NAMES = Set(["fspb", "peakbg", "fs-peakbg", "peak-background", "fspb+ssvep"])

_loss_requires_peak_windows(loss_name::String) = lowercase(loss_name) in _PEAK_LOSS_NAMES

function _merge_peak_windows(windows::Vector{Tuple{Float64, Float64}})
    isempty(windows) && return windows
    sorted = sort(windows, by = w -> w[1])
    merged = Tuple{Float64, Float64}[]
    current = sorted[1]
    for w in Iterators.drop(sorted, 1)
        if w[1] <= current[2]
            current = (current[1], max(current[2], w[2]))
        else
            push!(merged, current)
            current = w
        end
    end
    push!(merged, current)
    return merged
end

function _mask_from_windows(freqs::AbstractVector{<:Real}, windows::Vector{Tuple{Float64, Float64}})
    mask = falses(length(freqs))
    for (lo, hi) in windows
        mask .|= (freqs .>= lo) .& (freqs .<= hi)
    end
    return mask
end

function _suggest_window_bins(width_hz::Float64, df::Float64, fallback::Int; min_bins::Int=3)
    if isnan(df) || df <= 0
        bins = fallback
    else
        bins = max(min_bins, Int(round(width_hz / df)))
    end
    bins = max(1, bins)
    return isodd(bins) ? bins : bins + 1
end

function _smooth_ma_vec(x::AbstractVector{<:Real}, w::Int)
    n = length(x)
    w <= 1 && return Float64.(x)
    y = Vector{Float64}(undef, n)
    half = w ÷ 2
    @inbounds for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        acc = 0.0
        for k in lo:hi
            acc += float(x[k])
        end
        y[i] = acc / (hi - lo + 1)
    end
    return y
end

function _running_quantile(x::AbstractVector{<:Real}, w::Int, q::Real)
    n = length(x)
    w <= 1 && return Float64.(x)
    y = Vector{Float64}(undef, n)
    half = w ÷ 2
    qf = clamp(float(q), 0.0, 1.0)
    @inbounds for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        tmp = Vector{Float64}(undef, hi - lo + 1)
        t = 1
        for k in lo:hi
            tmp[t] = float(x[k])
            t += 1
        end
        sort!(tmp)
        idx = clamp(Int(floor(1 + qf * (length(tmp) - 1))), 1, length(tmp))
        y[i] = tmp[idx]
    end
    return y
end

function detect_peak_windows(freqs::AbstractVector{<:Real},
                             logpsd::AbstractVector{<:Real},
                             ls::LossSettings;
                             fmin_peak_hz::Float64 = ls.peak_min_frequency_hz,
                             fmax_peak_hz::Float64 = ls.peak_max_frequency_hz)
    length(freqs) == length(logpsd) || return Tuple{Float64, Float64}[]
    n = length(freqs)
    (n < 5 || ls.max_peak_windows == 0) && return Tuple{Float64, Float64}[]

    f = Float64.(freqs)
    raw = Float64.(logpsd)
    has_log = psd_preproc_has_log(ls.psd_preproc)
    if has_log
        y = raw
    else
        if all(raw .> 0)
            y = log10.(max.(raw, eps(Float64)))
        else
            y = raw
        end
    end

    idx0 = searchsortedfirst(f, fmin_peak_hz)
    idx0 > length(f) && return Tuple{Float64, Float64}[]
    f = f[idx0:end]
    y = y[idx0:end]
    if isfinite(fmax_peak_hz)
        idx1 = searchsortedlast(f, fmax_peak_hz)
        idx1 <= 0 && return Tuple{Float64, Float64}[]
        f = f[1:idx1]
        y = y[1:idx1]
    end
    m = length(f)
    m < 5 && return Tuple{Float64, Float64}[]

    df = m > 1 ? median(diff(f)) : NaN
    smooth_bins = _suggest_window_bins(0.5, df, 7; min_bins=3)
    y_sm = _smooth_ma_vec(y, smooth_bins)

    base_bins = _suggest_window_bins(ls.peak_baseline_window_hz, df, 11; min_bins=7)
    baseline = _running_quantile(y_sm, base_bins, ls.peak_baseline_quantile)

    prom_thr = Float64(ls.peak_prominence_db)
    above = (y_sm .- baseline) .>= prom_thr
    candidates = Tuple{Float64, Float64, Float64}[]
    half_width = ls.peak_bandwidth_hz / 2
    i = 1
    while i <= m
        if above[i]
            j = i
            best_i = i
            best_val = y_sm[i]
            while j <= m && above[j]
                if y_sm[j] > best_val
                    best_val = y_sm[j]
                    best_i = j
                end
                j += 1
            end
            score = y_sm[best_i] - baseline[best_i]
            lo = max(f[1], f[best_i] - half_width)
            hi = min(f[end], f[best_i] + half_width)
            push!(candidates, (score, lo, hi))
            i = j
        else
            i += 1
        end
    end

    isempty(candidates) && return Tuple{Float64, Float64}[]
    sort!(candidates, by = c -> c[1], rev = true)
    selected = Tuple{Float64, Float64}[]
    limit = min(ls.max_peak_windows, length(candidates))
    for k in 1:limit
        _, lo, hi = candidates[k]
        push!(selected, (lo, hi))
    end
    return _merge_peak_windows(selected)
end

function build_broad_peak_metadata(freqs::AbstractVector{<:Real},
                             powers::AbstractVector{<:Real},
                             ls::LossSettings,
                             loss_name::String)
    (_loss_requires_peak_windows(loss_name) && ls.max_peak_windows > 0) || return nothing
    length(freqs) == length(powers) || return nothing
    windows = if ls.peak_detection_empty
        lo = max(minimum(freqs), ls.peak_min_frequency_hz)
        hi = min(maximum(freqs), ls.peak_max_frequency_hz)
        hi <= lo && return nothing
        [(lo, hi)]
    else
        ENMEEG.detect_peak_windows(freqs, powers, ls)
    end
    isempty(windows) && return nothing
    mask = _mask_from_windows(freqs, windows)
    return (peak_windows=windows,
            peak_mask=mask,
            reference_freqs=Float64.(freqs),
            reference_psd=Float64.(powers),
            n_windows=length(windows))
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
    nperseg_val, nfft_val, overlap_val = ENMEEG._resolve_welch_params(n, Float64(fs), ls, nothing, ENMEEG._default_overlap())
    freqs, data_psd = compute_welch_pow_spectrum(Float64.(data), fs; xlims=fspan, nperseg=nperseg_val, nfft=nfft_val, overlap=overlap_val)
    isempty(freqs) && return nothing

    hf_start = max(ls.fmax - 10.0, ls.fmin)
    mask = freqs .>= hf_start
    any(mask) || return nothing
    S_data = median(data_psd[mask])
    (isfinite(S_data) && S_data > 0) || return nothing

    unit = ENMEEG._measurement_noise_template(n, ls)   # deterministic N(0,1)
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

    sigma_init = ENMEEG.estimate_sigma_init(ts_data)
    sigma_floor = (fs === nothing) ? nothing : ENMEEG.estimate_sigma_floor(ts_data, Float64(fs), ls)
    candidates = Float64[]
    (sigma_init !== nothing && isfinite(sigma_init) && sigma_init > 0) && push!(candidates, sigma_init)
    (sigma_floor !== nothing && isfinite(sigma_floor) && sigma_floor > 0) && push!(candidates, sigma_floor)
    isempty(candidates) && return nothing

    sigma_guess = minimum(candidates)
    resolved_sigma = (sigma_floor !== nothing && isfinite(sigma_floor) && sigma_floor > 0) ? sigma_floor : sigma_guess
    ls.sigma_meas = resolved_sigma
    return (sigma_init=sigma_init, sigma_floor=sigma_floor, sigma_guess=sigma_guess, sigma_applied=resolved_sigma)
end

