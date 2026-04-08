"""
-------------
LOSSES - Module containing loss functions and 
         metrics used in optimization.
-------------

"""

# Main function that takes the loss function as an argument
"""
    get_metric_function(loss_name::String)::Function

Returns the raw metric function based on the specified loss/metric name.
This is the single source of truth for mapping loss names to metric functions.

# Arguments
- `loss_name`: Name of the loss/metric ("TS-MAE", "FS-MAE", "H-MAE", "FSPB", etc.)
              Case-insensitive, supports multiple aliases.

# Returns
- `metric_function`: The raw metric function (e.g., metric_fsmae, metric_tsmae)

# Supported names and aliases:
- TS-MAE, tsmae → metric_tsmae
- FS-MAE, fsmae → metric_fsmae
- FS-Corr, fscorr → metric_fscorr
- BP-MAE, bpmae → metric_bpmae
- H-MAE, hmae → metric_hmae
- FSPB, peakbg, fs-peakbg, peak-background → metric_peak_background
- SSVEP, harmonic, harmonics, ssvep-harmonics → metric_ssvep_harmonic
- FSPB+SSVEP, peakbg+ssvep, fspb_ssvep, fspb-ssvep → metric_fspb_plus_ssvep
- BANDPOWER-REL, bandpower-rel → metric_bandpower_rel
"""
function get_metric_function(loss_name::String)::Function
    lname = lowercase(loss_name)
    
    if lname in ("ts-mae", "tsmae")
        return metric_tsmae
    elseif lname in ("fs-mae", "fsmae", "mae")
        return metric_fsmae
    elseif lname in ("fs-corr", "fscorr")
        return metric_fscorr
    elseif lname in ("bp-mae", "bpmae")
        return metric_bpmae
    elseif lname in ("h-mae", "hmae")
        return metric_hmae
    elseif lname in ("fspb", "peakbg", "fs-peakbg", "peak-background")
        return metric_peak_background
    elseif lname in ("ssvep", "harmonic", "harmonics", "ssvep-harmonics")
        return metric_ssvep_harmonic
    elseif lname in ("fspb+ssvep", "peakbg+ssvep", "fspb_ssvep", "fspb-ssvep")
        return metric_fspb_plus_ssvep
    elseif lname in ("bandpower-rel", "bandpower_rel")
        return metric_bandpower_rel
    else
        @warn "Unknown loss/metric name: $loss_name. Defaulting to FS-MAE."
        return metric_fsmae
    end
end


"""
    get_loss_function(loss_name::String)::Function

Returns the appropriate loss function based on the specified loss name.
This wraps the metric function with compute_loss for use in optimization.

# Arguments
- `loss_name`: Name of the loss function ("TS-MAE", "FS-MAE", "H-MAE", etc.)

# Returns
- `loss_function`: Function that computes the specified loss with optimization checks
"""
function get_loss_function(loss_name::String)::Function
    lname = lowercase(loss_name)
    
    if lname in ("advi",)
        return loss_empty
    end
    
    # Get the metric function and wrap it with compute_loss
    metric_fn = get_metric_function(loss_name)
    return (x, p) -> compute_loss(x, p, metric_fn)
end


"""
    compute_loss(x, p, loss_fun::Function)

Generic loss computation framework that handles parameter updating, ODE solving, 
and error detection, then applies a specific metric function to calculate the final loss.

# Arguments
- `x`: Parameter vector
- `p`: Tuple containing (prob, simulation_settings, data, setter, diffcache, solver, freq_range_idx)
- `loss_fun`: Function that calculates the specific metric between true and predicted data

# Returns
- `loss`: Computed loss value
"""

function compute_loss(new_params, args, metric_fun::Function)
    prob          = args.prob
    data        = args.data
    setter        = args.setter
    all_params    = args.all_params
    tspan         = args.tspan
    loss_settings = args.loss_settings
    brain_source_idx = args.brain_source_idx
    solver        = args.solver
    solver_kwargs = args.solver_kwargs

    n_inits = length(prob.u0)
    n_param_block = length(new_params) - n_inits
    @assert n_param_block >= 0 "Decision vector smaller than expected"

    θ = new_params[1:n_param_block]
    iv = new_params[n_param_block + 1 : n_param_block + n_inits]
    sigma_effective = max(loss_settings.sigma_meas, 0.0)
    updated_all_params = setter(all_params, θ)
    new_prob = remake(prob; p=updated_all_params, u0=iv, tspan=tspan)
    sol = safe_solve(new_prob, solver; solver_kwargs=solver_kwargs)
    
    if !SciMLBase.successful_retcode(sol)
        return 1e9
    end

    # Discard initial transient, e.g. first 1–2 s =
    keep_idx = findall(t -> t >= (tspan[1] + 2), sol.t)
    if isempty(keep_idx)
        return 1e9  # Solver terminated before transient period ended
    end
    model_prediction = Vector(sol[brain_source_idx, keep_idx])

    # Apply measurement noise if using time-series loss
    if metric_fun === metric_tsmae && sigma_effective > 0
        apply_measurement_noise!(model_prediction, sigma_effective, loss_settings)
    end
    
    if any(!isfinite, model_prediction)
        return 1e9
    end

    # crude growth detection to bail out exploding trajectories before PSD computation
    n = length(model_prediction)
    w = min(200, max(50, n ÷ 5))
    if n ≥ 2w
        # Compute RMS directly without allocating abs_trace
        head_sum_sq = 0.0
        tail_sum_sq = 0.0
        @inbounds for i in 1:w
            head_sum_sq += model_prediction[i]^2
            tail_sum_sq += model_prediction[n - w + i]^2
        end
        head_rms = sqrt(head_sum_sq / w)
        tail_rms = sqrt(tail_sum_sq / w)
        if isfinite(head_rms) && head_rms > 0 && tail_rms > loss_settings.max_rms_growth * head_rms
            return 1e9
        end
    end

    # Check max absolute value for signal bounds
    max_abs = maximum(abs, model_prediction)
    if max_abs > loss_settings.max_abs_signal
        return 1e9
    end

    # do demean
    model_prediction .-= mean(model_prediction)

    loss = metric_fun(model_prediction, data, 1/solver_kwargs[:saveat], loss_settings)
    
    if !isfinite(loss)
        return 1e9
    end
    return loss
end


function loss_empty(new_params, args)
    return NaN  # Or some default high loss value like 1e9
end
"""
    metric_tsmae(model_prediction, true_data, simulation_settings, freq_range_idx)

Calculate time-series mean absolute error metric.

# Returns
- `loss`: Mean squared error in time domain
"""
function metric_tsmae(model_prediction, target::TargetPSD, fs, loss_settings)
    observed = target.signal
    n = min(length(observed), length(model_prediction))
    n == 0 && return 1e9
    return mean(abs.(observed[1:n] .- model_prediction[1:n]))
end

"""
    metric_fsmae(model_prediction, true_data, simulation_settings, freqs)

Calculate frequency-spectrum mean absolute error metric.

# Returns
- `loss`: Mean absolute error in log frequency domain
"""
function metric_fsmae(model_prediction, target::TargetPSD, fs, loss_settings)
    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, target.freqs)
    return mean(abs.(target.powers .- aligned_model))
end


"""
    metric_fscorr(model_prediction, true_data, simulation_settings, freqs)

Calculate frequency-spectrum mean absolute error metric.

# Returns
- `loss`: 1 - correlation coefficient in log frequency domain
"""
function metric_fscorr(model_prediction, target::TargetPSD, fs, loss_settings)
    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, target.freqs)
    return 1 - cor(aligned_model, target.powers)
end


"""
    metric_peak_background and its helper functions
Calculate peak-background frequency spectrum loss metric.
"""
function _interpolate_psd(freqs_src::AbstractVector{<:Real}, values_src::AbstractVector{<:Real}, freqs_target::AbstractVector{<:Real})
    length(freqs_src) == length(values_src) || error("Source PSD arrays must have the same length.")
    isempty(freqs_target) && return Float64[]
    result = Vector{Float64}(undef, length(freqs_target))
    j = 1
    src_last = length(freqs_src)
    for (i, f) in enumerate(freqs_target)
        if f <= freqs_src[1]
            result[i] = Float64(values_src[1])
            continue
        elseif f >= freqs_src[end]
            result[i] = Float64(values_src[end])
            continue
        end
        while j < src_last && freqs_src[j + 1] < f
            j += 1
        end
        span = freqs_src[j + 1] - freqs_src[j]
        if span <= eps(Float64)
            result[i] = Float64(values_src[j])
        else
            α = (f - freqs_src[j]) / span
            result[i] = (1 - α) * Float64(values_src[j]) + α * Float64(values_src[j + 1])
        end
    end
    return result
end

function compute_peak_score(model_psd::AbstractVector{<:Real}, reference_psd::AbstractVector{<:Real}, peak_mask::AbstractVector{Bool})
    length(model_psd) == length(reference_psd) == length(peak_mask) || error("Peak score inputs must have matching lengths.")
    any(peak_mask) || return 0.0
    diffs = abs.(Float64.(model_psd[peak_mask]) .- Float64.(reference_psd[peak_mask]))
    return mean(diffs)
end

function compute_background_score(model_psd::AbstractVector{<:Real}, reference_psd::AbstractVector{<:Real}, peak_mask::AbstractVector{Bool})
    length(model_psd) == length(reference_psd) == length(peak_mask) || error("Background score inputs must have matching lengths.")
    background_mask = .!peak_mask
    any(background_mask) || return 0.0
    diffs = abs.(Float64.(model_psd[background_mask]) .- Float64.(reference_psd[background_mask]))
    return mean(diffs)
end

function metric_peak_background(model_prediction, target::TargetPSD, fs, loss_settings::LossSettings)
    pm = target.peak_metadata
    pm === nothing && return metric_fsmae(model_prediction, target, fs, loss_settings)

    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, pm.reference_freqs)
    peak_mask = pm.peak_mask
    any(peak_mask) || return metric_fsmae(model_prediction, target, fs, loss_settings)

    peak_score = compute_peak_score(aligned_model, pm.reference_psd, peak_mask)
    background_score = compute_background_score(aligned_model, pm.reference_psd, peak_mask)
    return loss_settings.weight_fspb * peak_score + loss_settings.weight_background * background_score
end


function metric_r2(psd_model, psd_data)
    ss_res = sum((psd_data .- psd_model).^2)
    ss_tot = sum((psd_data .- mean(psd_data)).^2)
    return 1 - ss_res/ss_tot
end


function _harmonic_mask(freqs::AbstractVector{<:Real}, f0::Real, H::Int, bw::Real; fmin=0.0, fmax=Inf)
    mask = falses(length(freqs))
    @inbounds for h in 1:H
        fh = h * f0
        (fh < fmin || fh > fmax) && continue
        for (i, f) in pairs(freqs)
            if abs(f - fh) <= bw
                mask[i] = true
            end
        end
    end
    return mask
end

function metric_ssvep_harmonic(model_prediction, target::TargetPSD, fs, loss_settings::LossSettings)
    # Compute model PSD using your existing pipeline
    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, target.freqs)

    f0 = loss_settings.ssvep_stim_freq_hz          # e.g. 5.45
    H  = loss_settings.ssvep_n_harmonics           # e.g. 3
    bw = loss_settings.ssvep_bandwidth_hz          # e.g. 0.8
    decay = loss_settings.ssvep_harmonic_decay     # e.g. 0.7

    peak_mask = _harmonic_mask(target.freqs, f0, H, bw; fmin=loss_settings.fmin, fmax=loss_settings.fmax)

    # If mask is empty, fall back
    if !any(peak_mask)
        return metric_fsmae(model_prediction, target, fs, loss_settings)
    end

    # Peak score with harmonic weighting (optional but matches your settings)
    # Implement harmonic weights by weighting each harmonic window separately.
    peak_score = _ssvep_harmonic_weighted_mae(aligned_model, target.powers, target.freqs, peak_mask, f0, H, bw, decay)

    # Background score (everything else)
    background_score = compute_background_score(aligned_model, target.powers, peak_mask)

    w_ssvep = loss_settings.weight_ssvep
    w_bg = loss_settings.weight_background

    return w_ssvep * peak_score + w_bg * background_score
end

function _ssvep_harmonic_weighted_mae(model_psd, ref_psd, freqs, peak_mask, f0, H, bw, decay)
    num = 0.0
    den = 0.0
    peak_idx = findall(peak_mask)
    isempty(peak_idx) && return 0.0
    freqs_peak = freqs[peak_idx]
    for h in 1:H
        fh = h * f0
        w = decay^(h-1)
        idx_local = findall(f -> abs(f - fh) <= bw, freqs_peak)
        isempty(idx_local) && continue
        idx = peak_idx[idx_local]
        num += w * mean(abs.(Float64.(model_psd[idx]) .- Float64.(ref_psd[idx])))
        den += w
    end
    return den > 0 ? num/den : 0.0
end


function metric_fspb_plus_ssvep(model_prediction, target::TargetPSD, fs, loss_settings::LossSettings)
    # Optional global weights (defaults to 1)
    w_fspb  = loss_settings.weight_fspb
    w_ssvep = loss_settings.weight_ssvep
    w_bg = loss_settings.weight_background

    pm = target.peak_metadata
    pm === nothing && return metric_fsmae(model_prediction, target, fs, loss_settings)

    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, target.freqs)

    # Build union mask for background: fspb peaks + SSVEP harmonics
    fspb_mask = pm.peak_mask
    f0 = loss_settings.ssvep_stim_freq_hz
    H = loss_settings.ssvep_n_harmonics
    bw = loss_settings.ssvep_bandwidth_hz
    harmonic_mask = _harmonic_mask(target.freqs, f0, H, bw; fmin=loss_settings.fmin, fmax=loss_settings.fmax)
    union_mask = fspb_mask .| harmonic_mask

    total = 0.0
    if loss_settings.fspb_enabled
        fspb_peak = compute_peak_score(aligned_model, pm.reference_psd, fspb_mask)
        total += w_fspb * fspb_peak
    end
    if loss_settings.ssvep_enabled
        ssvep_peak = _ssvep_harmonic_weighted_mae(aligned_model, target.powers, target.freqs, harmonic_mask,
                                                f0, H, bw, loss_settings.ssvep_harmonic_decay)
        total += w_ssvep * ssvep_peak
    end
    if loss_settings.fspb_enabled || loss_settings.ssvep_enabled
        background_score = compute_background_score(aligned_model, pm.reference_psd, union_mask)
        total += w_bg * background_score
        # print all three scores for logging
        #println("FSPB peak: $(loss_settings.fspb_enabled ? fspb_peak : "N/A"), SSVEP peak: $(loss_settings.ssvep_enabled ? ssvep_peak : "N/A"), Background: $background_score")
        #println("Scores weighted: FSPB=$(loss_settings.fspb_enabled ? w_fspb * fspb_peak : "N/A"), SSVEP=$(loss_settings.ssvep_enabled ? w_ssvep * ssvep_peak : "N/A"), Background=$(w_bg * background_score)")
        return total
    end

    # if both disabled, fall back to fsmae to avoid zero-loss bugs
    return metric_fsmae(model_prediction, target, fs, loss_settings)
end


"""
    metric_bandpower_rel(model_prediction, true_data, simulation_settings, freqs)

Loss based on relative band powers (theta/alpha/beta…).

1. Compute Welch PSD for each variable in data and model.
2. Integrate power in a set of frequency bands.
3. Normalize each band by total power (relative_total).
4. Return mean absolute error between data and model band-power vectors.

By default uses canonical EEG bands:
    θ: 4–8 Hz
    α: 8–13 Hz
    β_low: 13–20 Hz
    β_high: 20–30 Hz

If `simulation_settings` has a field `bands::Vector{Tuple{Float64,Float64}}`,
those are used instead.
"""
function metric_bandpower_rel(model_prediction::Matrix{Float64},
                              target_rel_bands::Dict{String,Float64},
                              simulation_settings,
                              freqs)
    # model_prediction: T × n_vars (you used Array(sol)')
    fs = 1 / simulation_settings.saveat

    # Only 1 variable for Duffing now, but code works for n_vars > 1
    model_ts = model_prediction[1, :]

    # Compute PSD in the same way as for data
    freqs_model, powers_model = compute_welch_pow_spectrum(model_ts, fs; window_type=hanning, xlims=(1., 48.))
    smoothed_model = smooth_power_spectrum(powers_model; method="savitzky_golay", window_size=15, poly_order=3)
    rel_bands_model = extract_frequency_band_powers(freqs_model, smoothed_model; normalize_method="relative_total")

    # Make sure we use the same band keys
    keys_target = collect(keys(target_rel_bands))
    diffs = Float64[]
    for k in keys_target
        push!(diffs, abs(rel_bands_model[k] - target_rel_bands[k]))
    end

    return sum(diffs)
end


"""
    metric_hmae(model_prediction, true_data, simulation_settings, freq_range_idx)

Calculate harmonic-focused mean absolute error metric.

# Returns
- `loss`: Weighted harmonic-focused mean absolute error in the frequency domain
"""
function metric_hmae(model_prediction, data_observed, simulation_settings, freq_range_idx)
    fs = 1/simulation_settings.saveat
    nfft = 2^nextpow(2, size(data_observed, 1))
    freq_resolution = fs/nfft
    freqs = freq_resolution .* (0:(nfft÷2))
    
    # Ensure consistent length for FFT
    if size(model_prediction, 1) > nfft
        model_prediction = model_prediction[1:nfft, :]
    elseif size(model_prediction, 1) < nfft
        padding = zeros(nfft - size(model_prediction, 1), size(model_prediction, 2))
        model_prediction = vcat(model_prediction, padding)
    end
    
    # Calculate observed data periodogram
    data_periodograms = DSP.Periodograms.welch_pgram.(eachrow(data_observed'), fs=fs, window=hanning, nfft=nfft)
    data_powers = hcat([periodogram.power[1:nfft÷2+1] for periodogram in data_periodograms]...)
    
    # Find harmonic weights (same code as in your original function)
    harmonic_weights = ones(length(freqs[freq_range_idx]))
    n_vars = size(data_observed, 2)
    
    for var_idx in 1:n_vars
        var_spectrum = data_powers[:, var_idx]
        peak_indices = findpeaks(var_spectrum[freq_range_idx])
        
        if !isempty(peak_indices)
            sorted_peaks = sort(peak_indices, by=i -> var_spectrum[freq_range_idx][i], rev=true)
            top_peaks = sorted_peaks[1:min(3, length(sorted_peaks))]
            
            for peak_idx in top_peaks
                peak_freq = freqs[freq_range_idx][peak_idx]
                harmonic_weights[peak_idx] = 5.0
                
                # Handle harmonics
                for harmonic in 2:5
                    harmonic_freq = peak_freq * harmonic
                    closest_idx = argmin(abs.(freqs[freq_range_idx] .- harmonic_freq))
                    window_size = 3
                    for w in max(1, closest_idx - window_size):min(length(harmonic_weights), closest_idx + window_size)
                        harmonic_weights[w] = max(harmonic_weights[w], 3.0)
                    end
                end
                
                # Handle subharmonics
                for divisor in 2:3
                    subharmonic_freq = peak_freq / divisor
                    if subharmonic_freq >= freqs[freq_range_idx[1]]
                        closest_idx = argmin(abs.(freqs[freq_range_idx] .- subharmonic_freq))
                        window_size = 3
                        for w in max(1, closest_idx - window_size):min(length(harmonic_weights), closest_idx + window_size)
                            harmonic_weights[w] = max(harmonic_weights[w], 2.0)
                        end
                    end
                end
            end
        end
    end
    
    # Calculate model periodogram
    model_periodograms = DSP.Periodograms.welch_pgram.(eachrow(model_prediction'), fs=fs, window=hanning, nfft=nfft)
    model_powers = hcat([periodogram.power[1:nfft÷2+1] for periodogram in model_periodograms]...)
    
    # Apply log transformation
    log_data = log10.(max.(data_powers[freq_range_idx, :], 1e-10))
    log_model = log10.(max.(model_powers[freq_range_idx, :], 1e-10))
    
    # Calculate weighted loss
    weighted_diffs = zeros(size(log_data))
    for var_idx in 1:n_vars
        weighted_diffs[:, var_idx] = harmonic_weights .* abs.(log_data[:, var_idx] .- log_model[:, var_idx])
    end
    
    return mean(weighted_diffs)
end


# Helper function to find peaks in a signal
function findpeaks(signal::Vector{Float64}; min_height::Float64=0.0, min_distance::Int=3)
    n = length(signal)
    peak_indices = Int[]
    
    # First and last points cannot be peaks
    for i in 2:n-1
        if signal[i] > min_height && 
           signal[i] > signal[i-1] && 
           signal[i] >= signal[i+1]
            # Found a potential peak
            if isempty(peak_indices) || (i - peak_indices[end]) >= min_distance
                push!(peak_indices, i)
            elseif signal[i] > signal[peak_indices[end]]
                # Replace previous peak if this one is higher and within min_distance
                peak_indices[end] = i
            end
        end
    end
    
    return peak_indices
end


## add new loss - calculation of CSD
function metric_csd(model_prediction, true_data, simulation_settings, freq_range_idx; fs=1000, nfft=256)
    n_channels = size(model_prediction, 1)
    model_csd = zeros(ComplexF64, n_channels, n_channels, nfft ÷ 2 + 1)
    for i in 1:n_channels
        for j in i:n_channels
            f, pxy = DSP.welch(model_prediction[i, :], model_prediction[j, :], fs=fs, nfft=nfft)
            model_csd[i, j, :] = pxy
            model_csd[j, i, :] = conj.(pxy)
        end
    end

    # Log-magnitude difference
    diff = log.(abs.(model_csd) .+ 1e-8) .- log.(abs.(true_data) .+ 1e-8)
    return sum(abs2, diff)

end


function metric_csd_weighted(model_prediction::Matrix{Float64}, observed_csd::Array{ComplexF64};
                           fs=1000, nfft=256, w_auto=1.0, w_cross=1.0)
    n_channels = size(model_prediction, 1)
    model_csd = zeros(ComplexF64, n_channels, n_channels, nfft ÷ 2 + 1)
    for i in 1:n_channels
        for j in i:n_channels
            f, pxy = DSP.welch(model_prediction[i, :], model_prediction[j, :], fs=fs, nfft=nfft)
            model_csd[i, j, :] = pxy
            model_csd[j, i, :] = conj.(pxy)
        end
    end

    n_channels = size(model_prediction, 1)
    total_loss = 0.0

    for i in 1:n_channels
        for j in 1:n_channels
            model_spec = log.(abs.(model_csd[i, j, :]) .+ 1e-8)
            obs_spec   = log.(abs.(observed_csd[i, j, :]) .+ 1e-8)
            diff = model_spec .- obs_spec

            if i == j
                total_loss += w_auto * sum(abs2, diff)
            else
                total_loss += w_cross * sum(abs2, diff)
            end
        end
    end

    return total_loss
end



# ================================
# == Measurement noise application ==
# ================================

# Draw a white-noise template (N(0,1)) of requested length
# If loss_noise_seed is nothing/null → use non-deterministic random noise
# If loss_noise_seed is an integer → use deterministic seeded noise
function _measurement_noise_template(len::Int, loss_settings::LossSettings)
    if len <= 0
        return Float64[]
    end
    if loss_settings.loss_noise_seed === nothing
        # No seed specified: use Julia's default random state (non-deterministic)
        vprint("Applying measurement noise, but loss_noise_seed is null: note that results will vary between runs/calls", level=1)
        return randn(len)
    else
        # Seed specified: use deterministic seeded RNG
        rng = Random.MersenneTwister(Int(loss_settings.loss_noise_seed))
        return randn(rng, len)
    end
end

function apply_measurement_noise!(data::AbstractVector, sigma, loss_settings::LossSettings)
    sigma === nothing && return data
    if Base.iszero(sigma)
        return data
    end
    len = length(data)
    len == 0 && return data
    noise = _measurement_noise_template(len, loss_settings)
    @inbounds @simd for i in 1:len
        data[i] += sigma * noise[i]
    end
    return data
end
