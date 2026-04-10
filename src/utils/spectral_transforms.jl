using FFTW
using DSP

"""
    SpectralTransforms Module

This module provides comprehensive tools for spectral analysis of time series data, 
particularly suited for EEG signal processing. It includes methods for computing 
various time-frequency representations, power spectral density estimation, 
normalization, and frequency band analysis.

## Main Methods

### Power Spectral Density (PSD)
- `compute_welch_pow_spectrum()`: Single-signal Welch PSD estimation
- `compute_welch_pow_spectra()`: Multi-signal Welch PSD estimation for DataFrames
- `calculate_spectrum()`: Complete spectral calculation with optional normalization
- `calculate_spectra()`: Process multiple signals from DataFrames

### Time-Frequency Representations
- `compute_cwt()`: Continuous Wavelet Transform using Morlet wavelets
- `compute_stft()`: Short-Time Fourier Transform with customizable parameters

### Normalization Methods
- `normalize_power_spectrum()`: Single signal normalization (db, relative, zscore, minmax, percent, log, log10, relative_db)
- `normalize_power_spectra()`: Multi-signal normalization for DataFrames

### Frequency Band Analysis
- `extract_frequency_band_powers()`: Extract power in standard EEG frequency bands (single or multi-signal)
- `calculate_band_relative_powers()`: Calculate relative power across frequency bands

### Smoothing
- `smooth_tf_powers()`: Apply smoothing to time-frequency representations (gaussian, moving_avg)

## Typical Workflow

1. Load time series data into a DataFrame with 'time' column
2. Calculate power spectra using `compute_welch_pow_spectra()`
3. Optionally normalize using `normalize_power_spectra()`
4. Extract frequency band powers using `extract_frequency_band_powers()`
5. For time-frequency analysis, use `compute_stft()` or `compute_cwt()`
"""

mutable struct WelchWorkspace
    window::Vector{Float64}
    segment::Vector{Float64}
    plan::FFTW.rFFTWPlan{Float64}
    fft_output::Vector{ComplexF64}
    accum::Vector{Float64}
    freqs_full::Vector{Float64}
    psd_full::Vector{Float64}
    freq_subset::Vector{Float64}
    psd_subset::Vector{Float64}
    nperseg::Int
    nfft::Int
    fs::Float64
    overlap::Float64
    hop::Int
    window_norm::Float64
end

mutable struct SpectrumWorkspace
    welch::WelchWorkspace
    scratch::Vector{Float64}
end

function WelchWorkspace(nperseg::Int, nfft::Int, fs::Real; window_type::Function=DSP.hanning, overlap::Float64=0.5)
    overlap = clamp(overlap, 0.0, 0.99)
    nperseg <= 0 && error("nperseg must be positive")
    nfft <= 0 && error("nfft must be positive")
    nfft < nperseg && error("nfft must be >= nperseg")
    window = window_type(nperseg)
    segment = zeros(Float64, nfft)
    plan = FFTW.plan_rfft(segment; flags=FFTW.MEASURE)
    fft_output = Vector{ComplexF64}(undef, div(nfft, 2) + 1)
    accum = zeros(Float64, length(fft_output))
    freqs_full = collect(0:Float64(fs)/nfft:Float64(fs)/2)
    psd_full = zeros(Float64, length(fft_output))
    freq_subset = similar(freqs_full)
    psd_subset = similar(psd_full)
    hop = max(1, Int(round(nperseg * (1 - overlap))))
    window_norm = sum(abs2, window)
    return WelchWorkspace(window, segment, plan, fft_output, accum, freqs_full, psd_full, freq_subset, psd_subset,
                          nperseg, nfft, Float64(fs), overlap, hop, window_norm)
end

function SpectrumWorkspace(nperseg::Int, nfft::Int, fs::Real; window_type::Function=DSP.hanning, overlap::Float64=0.5)
    welch = WelchWorkspace(nperseg, nfft, fs; window_type=window_type, overlap=overlap)
    scratch = zeros(Float64, length(welch.psd_full))
    return SpectrumWorkspace(welch, scratch)
end

function _resize_welch_workspace!(ws::WelchWorkspace, nperseg::Int, nfft::Int, fs::Float64, window_type::Function, overlap::Float64)
    overlap = clamp(overlap, 0.0, 0.99)
    nperseg <= 0 && error("nperseg must be positive")
    nfft <= 0 && error("nfft must be positive")
    nfft < nperseg && error("nfft must be >= nperseg")
    ws.nperseg = nperseg
    ws.nfft = nfft
    ws.fs = fs
    ws.overlap = overlap
    ws.hop = max(1, Int(round(nperseg * (1 - overlap))))
    ws.window = window_type(nperseg)
    ws.segment = zeros(Float64, nfft)
    ws.plan = FFTW.plan_rfft(ws.segment; flags=FFTW.MEASURE)
    len_spec = div(nfft, 2) + 1
    ws.fft_output = Vector{ComplexF64}(undef, len_spec)
    ws.accum = zeros(Float64, len_spec)
    ws.freqs_full = collect(0:fs/nfft:fs/2)
    ws.psd_full = zeros(Float64, len_spec)
    ws.freq_subset = zeros(Float64, len_spec)
    ws.psd_subset = zeros(Float64, len_spec)
    ws.window_norm = sum(abs2, ws.window)
    return ws
end

function _ensure_welch_workspace(ws::Union{Nothing, WelchWorkspace}, nperseg::Int, nfft::Int, fs::Float64, window_type::Function, overlap::Float64)
    if ws === nothing
        return WelchWorkspace(nperseg, nfft, fs; window_type=window_type, overlap=overlap)
    elseif ws.nperseg != nperseg || ws.nfft != nfft || ws.fs != fs || ws.overlap != overlap
        return _resize_welch_workspace!(ws, nperseg, nfft, fs, window_type, overlap)
    end
    return ws
end

function _ensure_spectrum_workspace(workspace::Union{Nothing, SpectrumWorkspace}, nperseg::Int, nfft::Int, fs::Float64, window_type::Function, overlap::Float64)
    if workspace === nothing
        return SpectrumWorkspace(nperseg, nfft, fs; window_type=window_type, overlap=overlap)
    end
    _ensure_welch_workspace(workspace.welch, nperseg, nfft, fs, window_type, overlap)
    if length(workspace.scratch) != length(workspace.welch.psd_full)
        workspace.scratch = zeros(Float64, length(workspace.welch.psd_full))
    end
    return workspace
end

_default_overlap() = 0.5

const _IMAGEFILTERING_MODULE = Ref{Union{Module, Nothing}}(nothing)
const _IMAGEFILTERING_TRIED = Ref(false)

function _ensure_imagefiltering_module()
    if _IMAGEFILTERING_TRIED[]
        return _IMAGEFILTERING_MODULE[]
    end
    _IMAGEFILTERING_TRIED[] = true
    try
        mod = Base.require(:ImageFiltering)
        _IMAGEFILTERING_MODULE[] = mod
        return mod
    catch err
        _IMAGEFILTERING_MODULE[] = nothing
        return nothing
    end
end

function _require_imagefiltering(feature::String)
    mod = _ensure_imagefiltering_module()
    mod === nothing && error("$(feature) smoothing requires ImageFiltering.jl. Install it with `Pkg.add(\"ImageFiltering\")` and try again.")
    return mod
end

function _resolve_nfft(len::Int, nfft::Union{Int, Nothing})
    if nfft === nothing
        return max(64, min(len, 2048))
    else
        return nfft
    end
end

function _resolve_welch_params(len::Int, fs::Float64, loss_settings::Union{Nothing, LossSettings}, nfft::Union{Int, Nothing}, overlap::Float64)
    nperseg = min(len, 256)
    nfft_eff = _resolve_nfft(len, nfft)
    overlap_eff = overlap

    loss_settings === nothing && return nperseg, nfft_eff, overlap_eff

    if loss_settings.psd_welch_nperseg > 0
        nperseg = min(loss_settings.psd_welch_nperseg, len)
    elseif loss_settings.psd_welch_window_sec > 0 && fs > 0
        nperseg = Int(round(fs * loss_settings.psd_welch_window_sec))
        nperseg = clamp(nperseg, 64, len)
    end

    overlap_eff = loss_settings.psd_welch_overlap

    if loss_settings.psd_welch_nfft > 0
        nfft_eff = max(loss_settings.psd_welch_nfft, nperseg)
    else
        nfft_eff = max(_resolve_nfft(len, nfft), nperseg)
    end

    hop = max(1, Int(round(nperseg * (1 - overlap_eff))))
    nseg_est = 1 + max(0, (len - nperseg) ÷ hop)
    if nseg_est < 2
        overlap_eff = 0.0
    end

    return nperseg, nfft_eff, overlap_eff
end

function _select_frequency_range!(ws::WelchWorkspace, xlims::Union{Nothing, Tuple{Float64, Float64}})
    len_full = length(ws.freqs_full)
    if xlims === nothing
        @inbounds for i in 1:len_full
            ws.freq_subset[i] = ws.freqs_full[i]
            ws.psd_subset[i] = ws.psd_full[i]
        end
        return view(ws.freq_subset, 1:len_full), view(ws.psd_subset, 1:len_full)
    end
    fmin, fmax = xlims
    count = 0
    @inbounds for i in 1:len_full
        f = ws.freqs_full[i]
        if f >= fmin && f <= fmax
            count += 1
            ws.freq_subset[count] = f
            ws.psd_subset[count] = ws.psd_full[i]
        end
    end
    return view(ws.freq_subset, 1:count), view(ws.psd_subset, 1:count)
end

function _accumulate_segment!(ws::WelchWorkspace)
    FFTW.mul!(ws.fft_output, ws.plan, ws.segment)
    @inbounds for k in eachindex(ws.fft_output)
        ws.accum[k] += abs2(ws.fft_output[k])
    end
end

function _welch_psd!(ws::WelchWorkspace, data::AbstractVector{<:Real}, xlims::Union{Nothing, Tuple{Float64, Float64}})
    fill!(ws.accum, 0.0)
    len = length(data)
    len == 0 && return view(ws.freq_subset, 1:0), view(ws.psd_subset, 1:0)
    nfft = ws.nfft
    nperseg = ws.nperseg
    idx = 1
    n_segments = 0
    while idx <= len
        remaining = min(nperseg, len - idx + 1)
        copyto!(ws.segment, 1, data, idx, remaining)
        if remaining < nfft
            fill!(view(ws.segment, remaining+1:nfft), 0.0)
        end
        @inbounds @simd for k in 1:nperseg
            ws.segment[k] *= ws.window[k]
        end
        _accumulate_segment!(ws)
        n_segments += 1
        if remaining < nperseg
            break
        end
        idx += ws.hop
    end
    scale = n_segments == 0 ? 0.0 : 1.0 / (ws.window_norm * ws.fs * n_segments)
    len_psd = length(ws.psd_full)
    @inbounds for k in 1:len_psd
        ws.psd_full[k] = ws.accum[k] * scale
    end
    if len_psd > 2
        @inbounds for k in 2:len_psd-1
            ws.psd_full[k] *= 2
        end
    end
    return _select_frequency_range!(ws, xlims)
end

function calculate_spectra(dfs_sources::Vector{DataFrame}, 
                           normalization_method::String="")::Vector{DataFrame}

    dfs_sources_spect = Vector{DataFrame}()

    for df_source in dfs_sources
        df_source_spect = calculate_spectrum(df_source; normalization_method=normalization_method)
        push!(dfs_sources_spect, df_source_spect)
    end

    return dfs_sources_spect
end


function calculate_spectrum(df_source::DataFrame; normalization_method::String="")
    
    # Calculate sampling frequency from time data
    time_col = df_source.time
    fs = 1.0 / (time_col[2] - time_col[1])
    
    # Compute frequency domain data for all signals
    df_source_spect = compute_welch_pow_spectra(df_source, fs)

    # Normalize
    if normalization_method != ""
        df_source_spect_norm = normalize_power_spectra(df_source_spect; method=normalization_method)
    else
        df_source_spect_norm = df_source_spect
    end
    return df_source_spect_norm
end


###################################################################################
## (TIME) FREQ TRANSFORMS
###################################################################################

function compute_cwt(data::Vector{Float64}, fs::Float64; 
                     ω₀=6.0, min_freq=1.0, max_freq=fs/4, n_freqs=50)
    
    n = length(data)
    dt = 1.0 / fs
    t = (0:(n-1)) .* dt
    
    # Create logarithmically spaced frequencies between min_freq and max_freq
    freqs = exp.(range(log(min_freq), log(max_freq), length=n_freqs))
    
    # Prepare output matrix
    powers = zeros(n_freqs, n)
    
    # Calculate wavelet transform for each frequency
    for (i, freq) in enumerate(freqs)
        # Scale for this frequency
        scale = ω₀ / (2π * freq)
        
        # Create time-centered wavelet
        t_center = n*dt/2
        t_wavelet = t .- t_center
        
        # Create Morlet wavelet
        wavelet = @. exp(im * ω₀ * t_wavelet/scale) * exp(-(t_wavelet/scale)^2 / 2)
        
        # Normalize
        wavelet ./= norm(wavelet)
        
        # Compute convolution via FFT
        data_fft = fft(data)
        wavelet_fft = fft(wavelet)
        conv_result = ifft(data_fft .* wavelet_fft)
        
        # Store power for this frequency
        powers[i, :] = abs.(conv_result).^2
    end

    return freqs, powers
end

"""
    compute_stft(data::Vector{Float64}, fs::Float64; 
                window_size=256, overlap=0.75, 
                min_freq=1.0, max_freq=fs/4, n_freqs=50)

Compute Short-Time Fourier Transform (STFT) of the input data.

# Arguments
- `data`: Input time series data
- `fs`: Sampling frequency (Hz)
- `window_size`: Size of each time window (samples)
- `overlap`: Overlap between consecutive windows (fraction between 0 and 1)
- `min_freq`: Minimum frequency to analyze (Hz)
- `max_freq`: Maximum frequency to analyze (Hz)
- `n_freqs`: Number of frequency bins

# Returns
- `freqs`: Vector of frequencies (Hz)
- `powers`: Power matrix (frequencies × times)
"""
function compute_stft(data::Vector{Float64}, fs::Float64; 
                     window_size=256, overlap=0.75,
                     min_freq=1.0, max_freq=fs/4, n_freqs=50,
                     smoothing="none", time_sigma=1.0, freq_sigma=1.0,
                     time_window=3, freq_window=3)
    
    # Calculate hop size and prepare windows
    hop_size = round(Int, window_size * (1 - overlap))
    window_func = DSP.hanning(window_size)
    
    # Calculate the number of frames
    n_frames = 1 + div(length(data) - window_size, hop_size)
    
    # Create logarithmically spaced frequencies between min_freq and max_freq
    freqs = exp.(range(log(min_freq), log(max_freq), length=n_freqs))
    
    # Calculate frequency bin indices (for later extraction)
    fft_freqs = fftfreq(window_size, fs) |> fftshift
    freq_indices = Vector{Int}(undef, n_freqs)
    
    for (i, f) in enumerate(freqs)
        # Find closest frequency bin
        _, idx = findmin(abs.(fft_freqs .- f))
        freq_indices[i] = idx
    end
    
    # Prepare output matrix
    powers = zeros(n_freqs, n_frames)
    
    # Process each frame
    for frame in 1:n_frames
        # Extract frame data
        start_idx = (frame - 1) * hop_size + 1
        end_idx = min(start_idx + window_size - 1, length(data))
        
        # Apply window function
        if end_idx - start_idx + 1 == window_size
            frame_data = data[start_idx:end_idx] .* window_func
        else
            # Zero-pad if near the end
            frame_data = zeros(window_size)
            frame_data[1:(end_idx-start_idx+1)] = data[start_idx:end_idx] .* window_func[1:(end_idx-start_idx+1)]
        end
        
        # Compute FFT
        fft_result = fft(frame_data) |> fftshift
        
        # Extract powers at the desired frequencies
        for (i, idx) in enumerate(freq_indices)
            powers[i, frame] = abs2(fft_result[idx])
        end
    end

    # After calculating powers, apply smoothing
    if smoothing != "none"
        powers = smooth_tf_powers(powers; 
                                 method=smoothing,
                                 time_sigma=time_sigma, 
                                 freq_sigma=freq_sigma,
                                 time_window=time_window,
                                 freq_window=freq_window)
    end
    
    
    return freqs, powers, window_size, hop_size
end


function compute_welch_pow_spectrum(data::AbstractVector{<:Real}, fs::Real;
                                    window_type::Function=DSP.hanning,
                                    xlims::Union{Tuple{Float64, Float64}, Nothing}=nothing,
                                    nperseg::Union{Int, Nothing}=nothing,
                                    nfft::Union{Int, Nothing}=nothing,
                                    overlap::Float64=_default_overlap(),
                                    workspace::Union{Nothing, WelchWorkspace}=nothing)
    len = length(data)
    len == 0 && return Float64[], Float64[]
    fs_val = Float64(fs)
    nperseg_val = nperseg === nothing ? min(len, 256) : min(nperseg, len)
    nfft_val = max(_resolve_nfft(len, nfft), nperseg_val)
    ws = _ensure_welch_workspace(workspace, nperseg_val, nfft_val, fs_val, window_type, overlap)
    freq_view, psd_view = _welch_psd!(ws, data, xlims)
    return copy(freq_view), copy(psd_view)
end

"""
    compute_welch_pow_spectra(df::DataFrame, fs::Float64; 
                              signal_cols=nothing, window_type=DSP.hanning, xlims=nothing)

Compute Welch power spectra for multiple signals in a DataFrame. Welch's method returns 
power spectral density (PSD) — typically in units of V²/Hz (or arbitrary units if 
uncalibrated).

# Arguments
- `df`: DataFrame containing time series data
- `fs`: Sampling frequency (Hz)
- `signal_cols`: Vector of column names to process (if nothing, excludes :time column)
- `window_type`: Window function for Welch method
- `xlims`: Tuple (fmin, fmax) to limit frequency range, or nothing for full range

# Returns
- DataFrame with columns: signal, frequency, power
"""
function compute_welch_pow_spectra(df::DataFrame, fs::Float64; 
                                   signal_cols=nothing, window_type=DSP.hanning, xlims=nothing)
    
    # Get signal columns (exclude time column if not specified)
    if signal_cols === nothing
        signal_cols = filter(n -> n != "time", names(df))
    end
    
    fr_df = DataFrame()
    workspace = nothing
    resolved_nfft = nothing
    resolved_nperseg = nothing
    
    # Compute frequency domain data for each signal
    for (ic, col) in enumerate(signal_cols)
        signal_data = Vector{Float64}(df[!, col])
        if workspace === nothing
            resolved_nperseg = min(length(signal_data), 256)
            resolved_nfft = max(_resolve_nfft(length(signal_data), nothing), resolved_nperseg)
            workspace = WelchWorkspace(resolved_nperseg, resolved_nfft, fs; window_type=window_type, overlap=_default_overlap())
        end
        freqs, powers = compute_welch_pow_spectrum(signal_data, fs;
                                                   window_type=window_type,
                                                   xlims=xlims,
                                                   nperseg=resolved_nperseg,
                                                   nfft=resolved_nfft,
                                                   workspace=workspace)
        if ic == 1
            fr_df.freq = freqs
        end
        fr_df[!, col] = powers
    end
    
    return fr_df
end

###################################################################################
## PSD PREPROCESSING PIPELINE
###################################################################################

abstract type AbstractPSDPreprocessOp end

struct PSDNormalizeOp <: AbstractPSDPreprocessOp
    method::String
end

struct PSDLogOp <: AbstractPSDPreprocessOp
    base::Symbol
    eps::Float64
end

struct PSDOffsetOp <: AbstractPSDPreprocessOp
    reducer::Symbol
end

struct PSDSmoothOp <: AbstractPSDPreprocessOp
    method::String
    window_size::Int
    poly_order::Int
    sigma::Float64
end

struct PSDPipelineContext
    window_size::Int
    poly_order::Int
    rel_eps::Float64
    smooth_sigma::Float64
end

const _PSD_NORMALIZATION_TOKENS = Set(["relative", "relative_db", "zscore", "minmax", "percent", "db"])
const _PSD_OFFSET_MEAN_TOKENS = Set(["offset", "offsetmean", "offset_mean", "demean", "center"])
const _PSD_OFFSET_MEDIAN_TOKENS = Set(["offsetmedian", "offset_median", "median", "medoffset"])
const _PSD_SMOOTH_TOKEN_OVERRIDES = Dict(
    "savgol" => "savitzky_golay",
    "savitzky_golay" => "savitzky_golay",
    "savitzkygolay" => "savitzky_golay",
    "gaussian" => "gaussian",
    "movingavg" => "moving_avg",
    "moving_avg" => "moving_avg",
    "ma" => "moving_avg"
)

const _PSD_NORMALIZATION_ALIASES = Dict(
    "norm" => "relative",
    "normalize" => "relative"
)

const _PSD_SMOOTHING_ALIASES = Dict(
    "smooth" => "savitzky_golay",
    "smoothing" => "savitzky_golay"
)

function _psd_token_to_op(token::String, ctx::PSDPipelineContext)
    token == "" && return nothing
    if haskey(_PSD_NORMALIZATION_ALIASES, token)
        method = _PSD_NORMALIZATION_ALIASES[token]
        method == "none" && return nothing
        return PSDNormalizeOp(method)
    elseif token in _PSD_NORMALIZATION_TOKENS
        return PSDNormalizeOp(token)
    elseif token in ("log", "ln")
        return PSDLogOp(:log, ctx.rel_eps)
    elseif token == "log10"
        return PSDLogOp(:log10, ctx.rel_eps)
    elseif token == "log2"
        return PSDLogOp(:log2, ctx.rel_eps)
    elseif token in _PSD_OFFSET_MEAN_TOKENS
        return PSDOffsetOp(:mean)
    elseif token in _PSD_OFFSET_MEDIAN_TOKENS
        return PSDOffsetOp(:median)
    elseif haskey(_PSD_SMOOTHING_ALIASES, token)
        method = _PSD_SMOOTHING_ALIASES[token]
        method == "none" && return nothing
        return PSDSmoothOp(method, ctx.window_size, ctx.poly_order, ctx.smooth_sigma)
    elseif haskey(_PSD_SMOOTH_TOKEN_OVERRIDES, token)
        method = _PSD_SMOOTH_TOKEN_OVERRIDES[token]
        method == "none" && return nothing
        return PSDSmoothOp(method, ctx.window_size, ctx.poly_order, ctx.smooth_sigma)
    elseif token == "none" || token == "raw"
        return nothing
    else
        error("Unknown PSD preprocessing token '$token'.")
    end
end

function parse_psd_preproc_pipeline(spec::Union{Nothing, AbstractString}, ctx::PSDPipelineContext)
    spec === nothing && return AbstractPSDPreprocessOp[]
    tokens = _psd_preproc_tokens(spec)
    isempty(tokens) && return AbstractPSDPreprocessOp[]
    ops = AbstractPSDPreprocessOp[]
    for token in tokens
        op = _psd_token_to_op(token, ctx)
        op === nothing && continue
        push!(ops, op)
    end
    return ops
end

function psd_preproc_flags_from_spec(spec::Union{Nothing, AbstractString})
    tokens = _psd_preproc_tokens(spec)
    return _psd_preproc_tokens_flags(tokens)
end

function psd_preproc_has_log(spec::Union{Nothing, AbstractString})
    has_log = psd_preproc_flags_from_spec(spec)
    return has_log
end

function _psd_scratch_slice(ws::SpectrumWorkspace, len::Int)
    len <= 0 && return view(ws.scratch, 1:0)
    if length(ws.scratch) < len
        resize!(ws.scratch, len)
    end
    return view(ws.scratch, 1:len)
end

function _apply_psd_preproc_op(op::PSDNormalizeOp, psd::Vector{Float64}, freqs)
    op.method == "none" && return copy(psd)
    result = similar(psd)
    return _normalize_power_spectrum!(result, psd, op.method)
end

function _apply_psd_preproc_op(op::PSDLogOp, psd::Vector{Float64}, freqs)
    data = max.(psd, op.eps)
    if op.base == :log10
        return log10.(data)
    elseif op.base == :log2
        return log2.(data)
    else
        return log.(data)
    end
end

function _apply_psd_preproc_op(op::PSDOffsetOp, psd::Vector{Float64}, freqs)
    baseline = op.reducer == :median ? median(psd) : mean(psd)
    return psd .- baseline
end

function _apply_psd_preproc_op(op::PSDSmoothOp, psd::Vector{Float64}, freqs)
    return smooth_power_spectrum(psd; method=op.method, window_size=op.window_size, poly_order=op.poly_order, sigma=op.sigma)
end

function _apply_psd_preproc_op!(op::PSDNormalizeOp,
                                out::AbstractVector{Float64},
                                psd::AbstractVector{Float64},
                                freqs)
    _normalize_power_spectrum!(out, psd, op.method)
end

function _apply_psd_preproc_op!(op::PSDLogOp,
                                out::AbstractVector{Float64},
                                psd::AbstractVector{Float64},
                                freqs)
    eps_val = op.eps
    base = op.base
    log_fun = base === :log10 ? log10 : base === :log2 ? log2 : log
    @inbounds @simd for i in eachindex(psd)
        x = psd[i]
        x = x > eps_val ? x : eps_val
        out[i] = log_fun(x)
    end
    return out
end

function _apply_psd_preproc_op!(op::PSDOffsetOp,
                                out::AbstractVector{Float64},
                                psd::AbstractVector{Float64},
                                freqs)
    baseline = op.reducer == :median ? median(psd) : mean(psd)
    @inbounds @simd for i in eachindex(psd)
        out[i] = psd[i] - baseline
    end
    return out
end

function _apply_psd_preproc_op!(op::AbstractPSDPreprocessOp,
                                out::AbstractVector{Float64},
                                psd::AbstractVector{Float64},
                                freqs)
    tmp = _apply_psd_preproc_op(op, collect(psd), freqs)
    length(tmp) == length(out) || error("In-place PSD op produced length mismatch.")
    copyto!(out, tmp)
    return out
end

function run_psd_preproc_pipeline!(ws::SpectrumWorkspace,
                                   psd::Vector{Float64},
                                   freqs::Vector{Float64},
                                   ops::Vector{AbstractPSDPreprocessOp})
    (isempty(ops) || isempty(psd)) && return psd
    len = length(psd)
    active = psd
    spare = _psd_scratch_slice(ws, len)

    for op in ops
        if op isa PSDSmoothOp
            active = _apply_psd_preproc_op(op, active, freqs)
            new_len = length(active)
            if new_len != len
                len = new_len
                spare = _psd_scratch_slice(ws, len)
            end
            continue
        end

        _apply_psd_preproc_op!(op, spare, active, freqs)
        active, spare = spare, active
    end

    if active === psd
        return psd
    end
    length(psd) == length(active) || resize!(psd, length(active))
    copyto!(psd, active)
    return psd
end

function _parse_psd_preproc_pipeline!(spec::String)::Vector{AbstractPSDPreprocessOp}
    spec == "none" && return AbstractPSDPreprocessOp[]
    # Standard hardcoded parameters
    ctx = PSDPipelineContext(11, 2, 1e-12, 1.0)
    return parse_psd_preproc_pipeline(spec, ctx)
end

""" 
compute_preprocessed_welch_psd(
        data::Vector{Float64}, fs::Real;
        xlims::Union{Tuple{Float64,Float64},Nothing}=nothing,
        window_type::Function=DSP.hanning,
        nfft::Union{Int,Nothing}=2048,
        window_size::Int=11,
        poly_order::Int=3,
        rel_eps::Float64=1e-12,
    )

Convenience helper that:
1. Computes Welch PSD
2. Applies the requested preprocessing pipeline (normalization/logging/smoothing)

Set `preproc_pipeline` (e.g. "relative-log10-savgol" or "offset-log10") to explicitly
control the preprocessing order. If omitted, the raw PSD is returned unless
`loss_settings` supplies a pipeline string.

Returns `(freqs, smoothed_log_power)`.
"""
function compute_preprocessed_welch_psd(
    data::AbstractVector{<:Real},
    fs::Real;
    window_type::Function=DSP.hanning,
    xlims::Union{Tuple{Float64,Float64},Nothing}=(1., 48.),
    nfft::Union{Int,Nothing}=2048,
    window_size::Int=5,
    poly_order::Int=2,
    rel_eps::Float64=1e-12,
    smooth_sigma::Float64=1.0,
    preproc_pipeline::Union{Nothing, AbstractString}=nothing,
    loss_settings::Union{Nothing, LossSettings}=nothing,
    workspace::Union{Nothing, SpectrumWorkspace}=nothing,
    overlap::Float64=_default_overlap(),
)

    len = length(data)
    len == 0 && return Float64[], Float64[]
    fs_val = Float64(fs)
    nperseg_val, nfft_val, overlap_val = _resolve_welch_params(len, fs_val, loss_settings, nfft, overlap)

    fspan = xlims
    pipeline_spec = preproc_pipeline

    ws = workspace
    if loss_settings !== nothing
        fspan = (loss_settings.fmin, loss_settings.fmax)
        pipeline_spec === nothing && (pipeline_spec = loss_settings.psd_preproc)
    end

    ws = _ensure_spectrum_workspace(ws, nperseg_val, nfft_val, fs_val, window_type, overlap_val)

    freq_view, psd_view = _welch_psd!(ws.welch, data, fspan)
    freqs = copy(freq_view)
    psd = copy(psd_view)

    effective_spec = pipeline_spec
    if effective_spec === nothing
        effective_spec = loss_settings === nothing ? "none" : loss_settings.psd_preproc
    end
    canonical_spec = _canonicalize_psd_preproc_string(effective_spec)
    if canonical_spec == "none"
        return freqs, psd
    end

    ops = _parse_psd_preproc_pipeline!(canonical_spec)
    psd = run_psd_preproc_pipeline!(ws, psd, freqs, ops)

    return freqs, psd
end

###################################################################################
## SMOOTHING
###################################################################################

function smooth_power_spectrum(powers::Vector{Float64}; 
                              method="savitzky_golay", 
                              window_size=11, 
                              poly_order=3,
                              sigma=1.0)
      
    if method == "none"
        return powers
    elseif method == "savitzky_golay"
        # Savitzky-Golay filter
        # Need to add: using DSP
        return savitzky_golay(powers, window_size, poly_order).y  # 2nd order polynomial
    elseif method == "gaussian"
        img = _require_imagefiltering("Gaussian")
        σ = sigma
        return img.imfilter(powers, img.Kernel.gaussian(σ))
    elseif method == "moving_avg"
        # Simple moving average filter
        result = copy(powers)
        n = length(powers)
        
        for i in 1:n
            window_start = max(1, i - div(window_size, 2))
            window_end = min(n, i + div(window_size, 2))
            result[i] = mean(powers[window_start:window_end])
        end
        
        return result
    else
        error("Unknown smoothing method: $method")
    end
end


function smooth_tf_powers(powers::Matrix{Float64}; 
                          method="gaussian", 
                          time_sigma=1.0, 
                          freq_sigma=1.0,
                          time_window=3, 
                          freq_window=3)
    
    if method == "none"
        return powers
    elseif method == "gaussian"
        img = _require_imagefiltering("Gaussian")
        σ = (freq_sigma, time_sigma)  # (y, x) format
        return img.imfilter(powers, img.Kernel.gaussian(σ))
    elseif method == "moving_avg"
        # Simple moving average filter
        result = copy(powers)
        n_freqs, n_times = size(powers)
        
        # Apply moving average in frequency dimension
        if freq_window > 1
            for t in 1:n_times
                for f in 1:n_freqs
                    window_start = max(1, f - div(freq_window, 2))
                    window_end = min(n_freqs, f + div(freq_window, 2))
                    result[f, t] = mean(powers[window_start:window_end, t])
                end
            end
        end
        
        # Apply moving average in time dimension
        if time_window > 1
            temp = copy(result)
            for f in 1:n_freqs
                for t in 1:n_times
                    window_start = max(1, t - div(time_window, 2))
                    window_end = min(n_times, t + div(time_window, 2))
                    result[f, t] = mean(temp[f, window_start:window_end])
                end
            end
        end
        
        return result
    else
        error("Unknown smoothing method: $method")
    end
end

###################################################################################
## NORMALIZATION
###################################################################################

function _normalize_power_spectrum!(out::AbstractVector{Float64},
                                     powers::AbstractVector{Float64},
                                     method::String)
    if method == "none"
        copyto!(out, powers)
        return out
    elseif method == "db"
        eps_val = eps(Float64)
        @inbounds @simd for i in eachindex(powers)
            out[i] = 10.0 * log10(powers[i] + eps_val)
        end
    elseif method == "relative"
        total_power = sum(powers)
        scale = 1 / total_power
        @inbounds @simd for i in eachindex(powers)
            out[i] = powers[i] * scale
        end
    elseif method == "zscore"
        μ = mean(powers)
        σ = std(powers)
        scale = 1 / σ
        @inbounds @simd for i in eachindex(powers)
            out[i] = (powers[i] - μ) * scale
        end
    elseif method == "minmax"
        min_power = minimum(powers)
        max_power = maximum(powers)
        range_power = max_power - min_power
        scale = 1 / range_power
        @inbounds @simd for i in eachindex(powers)
            out[i] = (powers[i] - min_power) * scale
        end
    elseif method == "percent"
        total_power = sum(powers)
        scale = 100 / total_power
        @inbounds @simd for i in eachindex(powers)
            out[i] = powers[i] * scale
        end
    elseif method == "log"
        eps_val = eps(Float64)
        @inbounds @simd for i in eachindex(powers)
            out[i] = log(powers[i] + eps_val)
        end
    elseif method == "log10"
        eps_val = eps(Float64)
        @inbounds @simd for i in eachindex(powers)
            out[i] = log10(powers[i] + eps_val)
        end
    elseif method == "relative_db"
        total_power = sum(powers)
        if total_power == 0
            fill!(out, 0.0)
            return out
        end
        eps_val = eps(Float64)
        scale = 1 / total_power
        @inbounds @simd for i in eachindex(powers)
            out[i] = 10.0 * log10(powers[i] * scale + eps_val)
        end
    else
        error("Unknown normalization method: $method. Available methods: none, db, relative, zscore, minmax, percent, log, log10, relative_db")
    end
    return out
end

"""
    normalize_power_spectrum(powers::Vector{Float64}; method::String="db")

Normalize power spectrum data using various methods.

# Arguments
- `powers`: Vector of power values
- `method`: Normalization method
  - "none": No normalization
  - "db": Convert to decibels (10 * log10(power))
  - "relative": Normalize by total power (power / sum(power))
    - "zscore": Z-score normalization ((power - mean) / std)
    - "minmax": Min-max normalization to [0, 1]
    - "percent": Express as percentage of total power
    - "log": Natural logarithm transform
    - "log10": Base-10 logarithm transform
    - "relative_db": Decibels relative to total power

# Returns
- Normalized power vector
"""
function normalize_power_spectrum(powers::Vector{Float64}; method::String="db")
    if method == "none"
        return powers
    end
    result = similar(powers)
    return _normalize_power_spectrum!(result, powers, method)
end

"""
    normalize_power_spectra(fr_data::DataFrame; method::String="db", signal_cols=nothing)

Normalize power spectra for multiple signals in a DataFrame.

# Arguments
- `fr_data`: DataFrame with freq column and signal columns
- `method`: Normalization method (see normalize_power_spectrum for options)
- `signal_cols`: Vector of column names to normalize (if nothing, excludes freq column)

# Returns
- DataFrame with normalized power values
"""
function normalize_power_spectra(fr_data::DataFrame; method::String="db", signal_cols=nothing)
    # Get signal columns (exclude freq column if not specified)
    if signal_cols === nothing
        signal_cols = filter(n -> n != "freq", names(fr_data))
    end
    
    # Create copy of the DataFrame
    normalized_df = copy(fr_data)
    
    # Normalize each signal column
    for col in signal_cols
        normalized_df[!, col] = normalize_power_spectrum(fr_data[!, col]; method=method)
    end
    
    return normalized_df
end

###################################################################################
## FREQ BANDS
###################################################################################

"""
    calculate_band_relative_powers(band_powers::Dict{String, Float64}, total_power::Float64; 
                                  normalize_method::String="relative_total")

Calculate relative powers for frequency bands based on normalization method.

# Arguments
- `band_powers`: Dictionary of band names and their absolute power values
- `total_power`: Total power across all frequencies (for relative_total method)
- `normalize_method`: Normalization method

# Returns
- Dictionary of band names and (absolute_power, relative_power) tuples
"""
function calculate_band_relative_powers(band_powers::Dict{String, Float64}, total_power::Float64; 
                                       normalize_method::String="relative_total")
    
    results = Dict{String, Float64}()
    
    if normalize_method == "relative_total"
        # Each band as percentage of total power across all frequencies
        for (band_name, band_power) in band_powers
            relative_power = (band_power / total_power) * 100
            results[band_name] = relative_power
        end
        
    elseif normalize_method == "relative_bands"
        # Each band as percentage of sum of all band powers
        total_band_power = sum(values(band_powers))
        for (band_name, band_power) in band_powers
            relative_power = total_band_power > 0 ? (band_power / total_band_power) * 100 : 0.0
            results[band_name] = relative_power
        end
        
    elseif normalize_method == "absolute"
        # No normalization - just raw power values or if it was normalized already
        for (band_name, band_power) in band_powers
            results[band_name] = band_power
        end
        
    else
        error("Unknown normalization method: $normalize_method. Available: relative_total, relative_bands, absolute")
    end
    
    return results
end

"""
    extract_frequency_bands(freqs::Vector{Float64}, powers::Vector{Float64}; 
                           bands::Dict{String, Tuple{Float64, Float64}}=nothing,
                           normalize_method::String="relative_total")

Extract power in specific frequency bands and calculate relative power for a single signal.
"""
function extract_frequency_band_powers(freqs::Vector{Float64}, powers::Vector{Float64}; 
                                bands::Union{Dict{String, Tuple{Float64, Float64}}, Nothing}=nothing,
                                normalize_method::String="relative_total")
    
    # Default EEG frequency bands if not provided
    if bands === nothing
        bands = get_freq_bands()
    end
    
    @assert length(freqs) == length(powers) "Frequency and power vectors must have the same length"
    
    # Calculate total power for normalization
    total_power = sum(powers)
    
    # Extract power for each band
    band_powers = Dict{String, Float64}()
    
    for (band_name, (fmin, fmax)) in bands
        # Find frequency indices within band
        band_indices = findall(f -> fmin <= f <= fmax, freqs)
        
        if !isempty(band_indices)
            # Sum power in this band (trapezoidal integration approximation)
            band_power = sum(powers[band_indices])
            band_powers[band_name] = band_power
        else
            vwarn("No frequencies found in band $band_name ($fmin-$fmax Hz)"; level=2)
            band_powers[band_name] = 0.0
        end
    end
    
    # Use the modular normalization function
    return calculate_band_relative_powers(band_powers, total_power; normalize_method=normalize_method)
end



function extract_frequency_band_powers(fr_data::DataFrame; 
                                bands::Union{Dict{String, Tuple{Float64, Float64}}, Nothing}=nothing,
                                normalize_method::String="relative_total")
    
    # Default EEG frequency bands if not provided
    if bands === nothing
        bands = get_freq_bands()
        @info "Using default EEG frequency bands: $(keys(bands))"
    end
    
    # Get signal columns (exclude freq column)
    signal_cols = filter(n -> n != "freq", names(fr_data))
    
    # Initialize results DataFrame
    results = DataFrame(
        signal = String[],
        band = String[],
        power = Float64[],
        relative_power = Float64[]
    )
    
    # Process each signal
    for signal_col in signal_cols
        freqs = fr_data.freq
        powers = fr_data[!, signal_col]
        
        # Calculate total power for normalization
        total_power = sum(powers)
        
        # Extract power for each band
        band_powers = Dict{String, Float64}()
        
        for (band_name, (fmin, fmax)) in bands
            # Find frequency indices within band
            band_indices = findall(f -> fmin <= f <= fmax, freqs)
            
            if !isempty(band_indices)
                # Sum power in this band (trapezoidal integration approximation)
                band_power = sum(powers[band_indices])
                band_powers[band_name] = band_power
            else
                vwarn("No frequencies found in band $band_name ($fmin-$fmax Hz) for signal $signal_col"; level=2)
                band_powers[band_name] = 0.0
            end
        end
        
        # Use the modular normalization function
        normalized_results = calculate_band_relative_powers(band_powers, total_power; normalize_method=normalize_method)
        
        # Add results to DataFrame
        for (band_name, (absolute_power, relative_power)) in normalized_results
            push!(results, (signal_col, band_name, absolute_power, relative_power))
        end
    end
    
    return results
end


###################################################################################
## EXAMPLES
###################################################################################

"""
    Example Usage of Spectral Transform Functions

This section demonstrates typical use cases for spectral analysis.
"""

# Example 1: Basic Welch Power Spectrum Calculation
# ================================================
# Compute power spectral density for a single signal using Welch's method
#
# data = randn(1000)  # 1000 samples of random data
# fs = 100.0  # Sampling frequency: 100 Hz
# freqs, powers = compute_welch_pow_spectrum(data, fs; xlims=(1.0, 50.0))
#
# This returns frequencies (1-50 Hz) and corresponding power values


# Example 2: Multi-Signal DataFrame Processing
# =============================================
# Process multiple signals from a DataFrame with normalization
#
# df = DataFrame(
#     time = range(0, 10, length=1000),
#     signal_1 = randn(1000),
#     signal_2 = randn(1000)
# )
# fs = 100.0
# df_spectra = compute_welch_pow_spectra(df, fs)
# df_spectra_db = normalize_power_spectra(df_spectra; method="db")
#
# Result: DataFrame with 'freq' column and normalized power for each signal


# Example 3: Frequency Band Power Analysis (Single Signal)
# =========================================================
# Extract power in standard EEG bands (delta, theta, alpha, beta, gamma)
#
# freqs, powers = compute_welch_pow_spectrum(data, 100.0)
# band_results = extract_frequency_band_powers(freqs, powers; 
#                                             normalize_method="relative_total")
#
# Result: Dictionary with bands as keys, (absolute_power, relative_power) as values
# Example output: Dict("delta" => (0.5, 15.2), "theta" => (1.2, 34.1), ...)


# Example 4: Frequency Band Analysis (Multi-Signal DataFrame)
# ============================================================
# Process multiple signals and extract band powers
#
# df_spectra = compute_welch_pow_spectra(df, 100.0)
# band_results = extract_frequency_band_powers(df_spectra; 
#                                             normalize_method="relative_bands")
#
# Result: DataFrame with columns [signal, band, power, relative_power]
#         One row per (signal, band) combination


# Example 5: Short-Time Fourier Transform (STFT)
# ===============================================
# Compute time-frequency representation with smoothing
#
# data = randn(5000)  # 5 seconds at 1000 Hz
# fs = 1000.0
# freqs, powers, wsize, hop = compute_stft(data, fs; 
#                                          window_size=512,
#                                          overlap=0.75,
#                                          min_freq=1.0,
#                                          max_freq=100.0,
#                                          smoothing="gaussian",
#                                          freq_sigma=2.0,
#                                          time_sigma=1.0)
#
# Result: freqs (vector), powers (n_freqs × n_times matrix)
# Each column represents a time window, each row a frequency


# Example 6: Continuous Wavelet Transform (CWT)
# ==============================================
# Compute CWT using Morlet wavelets
#
# data = randn(2000)
# fs = 100.0
# freqs, powers = compute_cwt(data, fs; 
#                            ω₀=6.0,
#                            min_freq=1.0,
#                            max_freq=40.0,
#                            n_freqs=40)
#
# Result: freqs (vector), powers (n_freqs × n_samples matrix)
# Good time resolution; frequency resolution depends on ω₀


# Example 7: Custom Frequency Bands
# ==================================
# Define and use custom frequency bands instead of standard EEG bands
#
# custom_bands = Dict(
#     "low_freq" => (1.0, 10.0),
#     "mid_freq" => (10.0, 30.0),
#     "high_freq" => (30.0, 100.0)
# )
# freqs, powers = compute_welch_pow_spectrum(data, 100.0)
# results = extract_frequency_band_powers(freqs, powers; 
#                                         bands=custom_bands,
#                                         normalize_method="relative_total")


# Example 8: Different Normalization Methods
# ===========================================
# Compare various normalization approaches
#
# data = randn(1000)
# fs = 100.0
# freqs, powers = compute_welch_pow_spectrum(data, fs)
#
# # All normalization methods:
# powers_db = normalize_power_spectrum(powers; method="db")           # Convert to dB
# powers_rel = normalize_power_spectrum(powers; method="relative")    # Normalize by total
# powers_z = normalize_power_spectrum(powers; method="zscore")        # Z-score
# powers_mm = normalize_power_spectrum(powers; method="minmax")       # Scale to [0,1]
# powers_pct = normalize_power_spectrum(powers; method="percent")     # Percentage
# powers_log = normalize_power_spectrum(powers; method="log")         # Natural log
# powers_log10 = normalize_power_spectrum(powers; method="log10")     # Log10
# powers_rdb = normalize_power_spectrum(powers; method="relative_db") # Relative dB


# Example 9: Complete Analysis Pipeline
# ======================================
# Full workflow from raw data to band analysis with visualization prep
#
# # Generate synthetic EEG-like data
# using Random
# Random.seed!(42)
# t = range(0, 10, length=5000)  # 10 seconds at 500 Hz
# signal = 0.5*sin.(2π*10*t) + 0.3*sin.(2π*20*t) + randn(length(t))*0.2  # 10Hz + 20Hz + noise
#
# df = DataFrame(time=t, eeg=signal)
# fs = 500.0
#
# # Step 1: Compute power spectrum
# df_psd = compute_welch_pow_spectra(df, fs; xlims=(1.0, 100.0))
#
# # Step 2: Normalize to dB scale
# df_psd_db = normalize_power_spectra(df_psd; method="db")
#
# # Step 3: Extract band powers
# band_powers = extract_frequency_band_powers(df_psd; normalize_method="relative_total")
#
# # Now ready for visualization or further analysis


# Example 10: Time-Frequency Analysis with Custom Parameters
# ===========================================================
# Advanced STFT with fine control
#
# data = randn(10000)
# fs = 1000.0
#
# freqs, powers, ws, hs = compute_stft(data, fs;
#                                      window_size=1024,      # 1024-point windows
#                                      overlap=0.875,          # 87.5% overlap
#                                      min_freq=0.5,
#                                      max_freq=200.0,
#                                      n_freqs=100,
#                                      smoothing="moving_avg", # Use moving average
#                                      time_window=5,
#                                      freq_window=3)
#
# # Powers is 100 × n_frames matrix
# # Each column is a time snapshot, each row is a frequency
