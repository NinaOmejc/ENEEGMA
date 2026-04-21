using FFTW
using DSP

"""
    SpectralTransforms Module

Spectral analysis tools for power spectral density (PSD) computation with flexible 
preprocessing pipelines. Designed for EEG and neural signal analysis.

## Main Methods

### Power Spectral Density (PSD)
- `compute_welch_psd()`: Single-signal Welch PSD estimation
- `compute_preprocessed_welch_psd()`: Welch PSD with preprocessing pipeline (normalization, log transform, smoothing)
- `compute_noisy_preprocessed_welch_psd()`: Welch PSD with preprocessing and noise averaging

### PSD Preprocessing Pipeline
- Normalization methods: relative, db, zscore, minmax, percent, log, log10, relative_db
- Log transforms: log, log2, log10
- Smoothing methods: savitzky_golay, gaussian, moving_avg
- Offset reduction: mean, median
- Extensible token-based pipeline syntax (e.g., "relative-log10-savgol")

### Normalization (deprecated - use preprocessing pipeline instead)
- `normalize_power_spectrum()`: Single signal normalization
- `normalize_power_spectra()`: Multi-signal normalization for DataFrames

## Typical Workflow

1. Load time series data into a DataFrame
2. Compute preprocessed PSD: `compute_preprocessed_welch_psd(signal, fs; loss_settings=loss_settings, data_settings=data_settings)`
3. Specify preprocessing via PSDSettings.preproc_pipeline or LossSettings.psd_preproc
4. Results include normalized/log-transformed power values
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

function _resolve_welch_params(len::Int, fs::Float64, data_settings::Union{Nothing, Any})
    # Get parameters from data_settings if available; otherwise use sensible defaults
    if data_settings !== nothing && hasproperty(data_settings, :psd)
        psd_settings = data_settings.psd
        
        # Determine nperseg
        if psd_settings.welch_nperseg > 0
            nperseg = min(psd_settings.welch_nperseg, len)
        elseif psd_settings.welch_window_sec > 0 && fs > 0
            nperseg = Int(round(fs * psd_settings.welch_window_sec))
            nperseg = clamp(nperseg, 64, len)
        else
            nperseg = min(len, 256)
        end

        # Determine nfft
        nfft_eff = if psd_settings.welch_nfft > 0
            max(psd_settings.welch_nfft, nperseg)
        else
            max(_resolve_nfft(len, nothing), nperseg)
        end
        
        # Get overlap from settings
        overlap_eff = psd_settings.welch_overlap
    else
        # Fallback defaults when data_settings is unavailable
        nperseg = min(len, 256)
        nfft_eff = _resolve_nfft(len, nothing)
        overlap_eff = 0.1
    end

    # Ensure at least 2 segments; otherwise disable overlap
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


function compute_welch_psd(data::AbstractVector{<:Real}, fs::Real;
                                    window_type::Function=DSP.hanning,
                                    xlims::Union{Tuple{Float64, Float64}, Nothing}=nothing,
                                    nperseg::Union{Int, Nothing}=nothing,
                                    nfft::Union{Int, Nothing}=nothing,
                                    overlap::Float64=0.1,
                                    workspace::Union{Nothing, WelchWorkspace}=nothing)
    len = length(data)
    len == 0 && return Float64[], Float64[]
    fs_val = Float64(fs)
    nperseg_val = nperseg === nothing ? min(len, 256) : min(nperseg, len)
    nfft_val = max(ENEEGMA._resolve_nfft(len, nfft), nperseg_val)
    ws = ENEEGMA._ensure_welch_workspace(workspace, nperseg_val, nfft_val, fs_val, window_type, overlap)
    freq_view, psd_view = ENEEGMA._welch_psd!(ws, data, xlims)
    return copy(freq_view), copy(psd_view)
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

function _run_psd_preproc_pipeline!(ws::SpectrumWorkspace,
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
    signal::AbstractVector{<:Real},
    fs::Real;
    window_type::Function=DSP.hanning,
    xlims::Union{Tuple{Float64,Float64},Nothing}=(1., 48.),
    nfft::Union{Int,Nothing}=2048,
    window_size::Int=5,
    poly_order::Int=2,
    rel_eps::Float64=1e-12,
    smooth_sigma::Float64=1.0,
    overlap::Float64=0.1,
    preproc_pipeline::Union{Nothing, AbstractString}=nothing,
    loss_settings::Union{Nothing, LossSettings}=nothing,
    data_settings::Union{Nothing, DataSettings}=nothing,
    workspace::Union{Nothing, SpectrumWorkspace}=nothing,
)

    len = length(signal)
    len == 0 && return Float64[], Float64[]
    fs_val = Float64(fs)
    nperseg_val, nfft_val, overlap_val = ENEEGMA._resolve_welch_params(len, fs_val, data_settings)

    fspan = xlims
    pipeline_spec = preproc_pipeline

    ctx_window_size = window_size
    ctx_poly_order = poly_order
    ctx_rel_eps = rel_eps
    ctx_sigma = smooth_sigma

    ws = workspace
    
    # Extract PSD preprocessing settings from data_settings if available
    if data_settings !== nothing && hasproperty(data_settings, :psd)
        psd_settings = data_settings.psd
        ctx_window_size = psd_settings.window_size
        ctx_poly_order = psd_settings.smooth_poly_order
        ctx_rel_eps = psd_settings.rel_eps
        ctx_sigma = psd_settings.smooth_sigma
        pipeline_spec === nothing && (pipeline_spec = psd_settings.preproc_pipeline)
        if psd_settings.workspace !== nothing
            ws = psd_settings.workspace
        end
    end
    
    # Loss settings provides frequency range only
    if loss_settings !== nothing
        fspan = (loss_settings.fmin, loss_settings.fmax)
    end

    ws = ENEEGMA._ensure_spectrum_workspace(ws, nperseg_val, nfft_val, fs_val, window_type, overlap_val)
    if data_settings !== nothing && hasproperty(data_settings, :psd)
        data_settings.psd.workspace = ws
    end

    freq_view, psd_view = ENEEGMA._welch_psd!(ws.welch, signal, fspan)
    freqs = copy(freq_view)
    psd = copy(psd_view)

    effective_spec = pipeline_spec
    if effective_spec === nothing
        effective_spec = loss_settings === nothing ? "none" : loss_settings.psd_preproc
    end
    canonical_spec = ENEEGMA._canonicalize_psd_preproc_string(effective_spec)
    if canonical_spec == "none"
        return freqs, psd
    end

    ops = if loss_settings !== nothing && pipeline_spec === nothing
        ENEEGMA._get_losssettings_psd_ops!(loss_settings,
                                           canonical_spec,
                                           ctx_window_size,
                                           ctx_poly_order,
                                           ctx_rel_eps,
                                           ctx_sigma)
    else
        ctx = ENEEGMA.PSDPipelineContext(ctx_window_size, ctx_poly_order, ctx_rel_eps, ctx_sigma)
        ENEEGMA.parse_psd_preproc_pipeline(canonical_spec, ctx)
    end
    psd = ENEEGMA._run_psd_preproc_pipeline!(ws, psd, freqs, ops)

    return freqs, psd
end



# ================================
# == PSD computation with noise averaging ==
# ================================

function compute_noisy_preprocessed_welch_psd(model_prediction::AbstractVector{<:Real},
                                fs::Real,
                                loss_settings::LossSettings,
                                data_settings::DataSettings)
    reps = max(data_settings.psd.noise_avg_reps, 1)
    sigma_effective = max(data_settings.measurement_noise_std, 0.0)

    if sigma_effective <= 0
        return ENEEGMA.compute_preprocessed_welch_psd(model_prediction, fs; 
                                                      loss_settings=loss_settings,
                                                      data_settings=data_settings)
    end

    seed_value = data_settings.psd.noise_seed
    freqs = Float64[]
    accum = Float64[]
    for rep in 1:reps
        # Create RNG for this repetition (or use nothing for non-deterministic)
        rng = if seed_value === nothing
            nothing  # Non-deterministic
        else
            Random.MersenneTwister(Int(seed_value) + rep - 1)
        end
        
        noisy = Float64.(model_prediction)
        apply_measurement_noise!(noisy, sigma_effective, rng)
        freqs_rep, powers_rep = compute_preprocessed_welch_psd(noisy, fs; 
                                                               loss_settings=loss_settings,
                                                               data_settings=data_settings)
        if isempty(freqs)
            freqs = freqs_rep
            accum = zeros(length(powers_rep))
        end
        accum .+= powers_rep
    end
    return freqs, accum ./ reps
end

"""
    compute_psd_for_all_sources(df_sources::DataFrame, fs::Real; source_cols=nothing, kwargs...)

Compute PSDs for each source-signal column in `df_sources` by reusing
`compute_preprocessed_welch_psd`.

The input is expected to contain a `time` column plus one or more source columns,
for example the output of `extract_brain_sources`. The returned dictionary is keyed
by source column name and stores `(freqs, powers)` tuples for direct use in
`plot_simulation_results`.

# Arguments
- `df_sources::DataFrame`: DataFrame with `time` and source-signal columns
- `fs::Real`: Sampling frequency in Hz
- `source_cols`: Optional subset of source columns to process. Defaults to all
  columns except `time`.
- `kwargs...`: Passed directly to `compute_preprocessed_welch_psd`

# Returns
- `Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}`: PSD data for each source
"""
function compute_psd_for_all_sources(df_sources::DataFrame,
                                     fs::Real;
                                     source_cols=nothing,
                                     kwargs...)
    selected_cols = source_cols === nothing ?
        [String(col) for col in names(df_sources) if String(col) != "time"] :
        String.(source_cols)

    psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    for col in selected_cols
        freqs, powers = compute_preprocessed_welch_psd(df_sources[!, Symbol(col)], fs; kwargs...)
        psd_dict[col] = (freqs, powers)
    end

    return psd_dict
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


