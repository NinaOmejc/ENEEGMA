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
3. Specify preprocessing via `DataSettings.psd.preproc_pipeline` or the `preproc_pipeline` keyword
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

function _apply_psd_preproc_op(op::PSDSmoothOp, psd::AbstractVector{Float64}, freqs)
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

function _apply_psd_preproc_op!(op::PSDSmoothOp,
                                out::AbstractVector{Float64},
                                psd::AbstractVector{Float64},
                                freqs)
    smooth_power_spectrum!(out, psd; method=op.method, window_size=op.window_size, poly_order=op.poly_order, sigma=op.sigma)
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
    active = psd
    spare = _psd_scratch_slice(ws, length(psd))

    for op in ops
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
control the preprocessing order. If omitted, `data_settings.psd.preproc_pipeline`
is used when available; otherwise the raw PSD is returned.

Returns `(freqs, smoothed_log_power)`.
"""
function compute_preprocessed_welch_psd(
    signal::AbstractVector{<:Real},
    fs::Real;
    window_type::Function=DSP.hanning,
    xlims::Union{Tuple{Float64,Float64},Nothing}=(1., 48.),
    window_size::Int=5,
    poly_order::Int=2,
    rel_eps::Float64=1e-12,
    smooth_sigma::Float64=1.0,
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
        ctx_window_size = psd_settings.smooth_savgol_window_size
        ctx_poly_order = psd_settings.smooth_savgol_poly_order
        ctx_rel_eps = psd_settings.rel_eps
        ctx_sigma = psd_settings.smooth_gaussian_sigma
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

    effective_spec = pipeline_spec === nothing ? "none" : pipeline_spec
    canonical_spec = ENEEGMA._canonicalize_psd_preproc_string(effective_spec)
    if canonical_spec == "none"
        return freqs, psd
    end

    ctx = ENEEGMA.PSDPipelineContext(ctx_window_size, ctx_poly_order, ctx_rel_eps, ctx_sigma)
    ops = ENEEGMA.parse_psd_preproc_pipeline(canonical_spec, ctx)
    psd = ENEEGMA._run_psd_preproc_pipeline!(ws, psd, freqs, ops)

    return freqs, psd
end



# ================================
# == PSD computation with noise averaging ==
# ================================

function compute_noisy_preprocessed_welch_psd(model_prediction::AbstractVector{<:Real},
                                fs::Real,
                                loss_settings::LossSettings,
                                data_settings::DataSettings;
                                measurement_noise_std::Union{Real, Nothing}=nothing)
    reps = max(data_settings.psd.noise_avg_reps, 1)
    sigma_effective = measurement_noise_std === nothing ? 0.0 : max(Float64(measurement_noise_std), 0.0)

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
        ENEEGMA.apply_measurement_noise!(noisy, sigma_effective, rng)
        freqs_rep, powers_rep = ENEEGMA.compute_preprocessed_welch_psd(noisy, fs; 
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

function compute_noisy_preprocessed_welch_psd(model_prediction::AbstractVector{<:Real},
                                fs::Real,
                                loss_settings::LossSettings,
                                data_settings::DataSettings,
                                node_info::NodeData)
    return compute_noisy_preprocessed_welch_psd(model_prediction,
                                                fs,
                                                loss_settings,
                                                data_settings;
                                                measurement_noise_std=node_info.measurement_noise_std)
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
                                     data_settings::Union{Nothing, DataSettings}=nothing,
                                     loss_settings::Union{Nothing, LossSettings}=nothing,)
    selected_cols = source_cols === nothing ?
        [String(col) for col in names(df_sources) if String(col) != "time"] :
        String.(source_cols)

    psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    for col in selected_cols
        freqs, powers = compute_preprocessed_welch_psd(df_sources[!, Symbol(col)], fs; 
                                                       data_settings=data_settings,
                                                       loss_settings=loss_settings)
        psd_dict[col] = (freqs, powers)
    end

    return psd_dict
end



###################################################################################
## SMOOTHING
###################################################################################

struct SavitzkyGolayPlan
    starts::Vector{Int}
    lengths::Vector{Int}
    weights::Matrix{Float64}
end

struct GaussianKernelPlan
    radius::Int
    kernel::Vector{Float64}
end

const _SMOOTH_PLAN_LOCK = ReentrantLock()
const _SGOLAY_PLAN_CACHE = Dict{Tuple{Int, Int, Int}, SavitzkyGolayPlan}()
const _GAUSSIAN_PLAN_CACHE = Dict{Float64, GaussianKernelPlan}()

function _smooth_power_spectrum_legacy(powers::AbstractVector{Float64}; 
                              method="savitzky_golay", 
                              window_size=11, 
                              poly_order=3,
                              sigma=1.0)
      
    if method == "none"
        return powers
    elseif method == "savitzky_golay"
        return _savitzky_golay_smooth(powers, window_size, poly_order)
    elseif method == "gaussian"
        return _gaussian_smooth(powers, sigma)
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

function _gaussian_smooth_legacy(powers::AbstractVector{Float64}, sigma::Real)
    n = length(powers)
    n == 0 && return Float64[]
    sigma >= 0 || error("Gaussian smoothing requires sigma >= 0, got $sigma.")
    sigma == 0 && return copy(powers)

    radius = max(1, ceil(Int, 3 * sigma))
    offsets = -radius:radius
    kernel = exp.(-0.5 .* (Float64.(offsets) ./ Float64(sigma)).^2)
    kernel ./= sum(kernel)

    result = similar(powers, Float64)
    lo = firstindex(powers)
    hi = lastindex(powers)

    for i in eachindex(powers)
        acc = 0.0
        for (k, offset) in enumerate(offsets)
            idx = clamp(i + offset, lo, hi)
            acc += kernel[k] * powers[idx]
        end
        result[i] = acc
    end

    return result
end

function _savitzky_golay_smooth_legacy(powers::AbstractVector{Float64},
                                window_size::Int,
                                poly_order::Int)
    n = length(powers)
    n == 0 && return Float64[]
    window_size >= 1 || error("Savitzky-Golay smoothing requires window_size >= 1.")
    isodd(window_size) || error("Savitzky-Golay smoothing requires an odd window_size, got $window_size.")
    poly_order >= 0 || error("Savitzky-Golay smoothing requires poly_order >= 0.")
    window_size > poly_order || error("Savitzky-Golay smoothing requires window_size > poly_order.")

    result = similar(powers, Float64)
    half_window = div(window_size, 2)

    for i in eachindex(powers)
        left = max(firstindex(powers), i - half_window)
        right = min(lastindex(powers), i + half_window)
        idxs = left:right
        local_x = Float64[j - i for j in idxs]
        order = min(poly_order, length(local_x) - 1)
        vand = hcat((local_x .^ k for k in 0:order)...)
        coeffs = vand \ Float64.(powers[idxs])
        result[i] = coeffs[1]
    end

    return result
end

function _validate_savgol_params(window_size::Int, poly_order::Int)
    window_size >= 1 || error("Savitzky-Golay smoothing requires window_size >= 1.")
    isodd(window_size) || error("Savitzky-Golay smoothing requires an odd window_size, got $window_size.")
    poly_order >= 0 || error("Savitzky-Golay smoothing requires poly_order >= 0.")
    window_size > poly_order || error("Savitzky-Golay smoothing requires window_size > poly_order.")
end

function _validate_gaussian_sigma(sigma::Real)
    sigma >= 0 || error("Gaussian smoothing requires sigma >= 0, got $sigma.")
end

function _gaussian_plan(sigma::Float64)
    lock(_SMOOTH_PLAN_LOCK) do
        get!(_GAUSSIAN_PLAN_CACHE, sigma) do
            radius = max(1, ceil(Int, 3 * sigma))
            offsets = collect(-radius:radius)
            kernel = exp.(-0.5 .* (offsets ./ sigma) .^ 2)
            kernel ./= sum(kernel)
            GaussianKernelPlan(radius, kernel)
        end
    end
end

function _savitzky_golay_plan(n::Int, window_size::Int, poly_order::Int)
    key = (n, window_size, poly_order)
    lock(_SMOOTH_PLAN_LOCK) do
        get!(_SGOLAY_PLAN_CACHE, key) do
            half_window = div(window_size, 2)
            max_len = min(n, window_size)
            starts = Vector{Int}(undef, n)
            lengths = Vector{Int}(undef, n)
            weights = zeros(Float64, n, max_len)

            for i in 1:n
                left = max(1, i - half_window)
                right = min(n, i + half_window)
                len_i = right - left + 1
                starts[i] = left
                lengths[i] = len_i

                order_i = min(poly_order, len_i - 1)
                vand = Matrix{Float64}(undef, len_i, order_i + 1)

                for row in 1:len_i
                    x = Float64(left + row - 1 - i)
                    term = 1.0
                    for col in 1:(order_i + 1)
                        vand[row, col] = term
                        term *= x
                    end
                end

                coeffs = (vand' * vand) \ vand'
                @views weights[i, 1:len_i] .= coeffs[1, :]
            end

            SavitzkyGolayPlan(starts, lengths, weights)
        end
    end
end

function _moving_average_smooth!(out::AbstractVector{Float64},
                                 powers::AbstractVector{Float64},
                                 window_size::Int)
    window_size >= 1 || error("Moving-average smoothing requires window_size >= 1.")
    n = length(powers)
    half_window = div(window_size, 2)

    @inbounds for i in eachindex(powers)
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        acc = 0.0
        for j in start_idx:end_idx
            acc += powers[j]
        end
        out[i] = acc / (end_idx - start_idx + 1)
    end
    return out
end

function _gaussian_smooth!(out::AbstractVector{Float64},
                           powers::AbstractVector{Float64},
                           sigma::Real)
    _validate_gaussian_sigma(sigma)
    sigma == 0 && return copyto!(out, powers)

    plan = _gaussian_plan(Float64(sigma))
    kernel = plan.kernel
    radius = plan.radius
    n = length(powers)
    last = length(kernel)
    center_start = radius + 1
    center_end = n - radius

    @inbounds begin
        for i in 1:min(n, radius)
            acc = 0.0
            for k in 1:last
                idx = i + k - radius - 1
                idx = idx < 1 ? 1 : (idx > n ? n : idx)
                acc = muladd(kernel[k], powers[idx], acc)
            end
            out[i] = acc
        end

        if center_start <= center_end
            for i in center_start:center_end
                acc = 0.0
                base = i - radius - 1
                @simd for k in 1:last
                    acc = muladd(kernel[k], powers[base + k], acc)
                end
                out[i] = acc
            end
        end

        for i in max(center_end + 1, radius + 1):n
            acc = 0.0
            for k in 1:last
                idx = i + k - radius - 1
                idx = idx < 1 ? 1 : (idx > n ? n : idx)
                acc = muladd(kernel[k], powers[idx], acc)
            end
            out[i] = acc
        end
    end
    return out
end

function _savitzky_golay_smooth!(out::AbstractVector{Float64},
                                 powers::AbstractVector{Float64},
                                 window_size::Int,
                                 poly_order::Int)
    _validate_savgol_params(window_size, poly_order)
    n = length(powers)
    n == 0 && return out

    plan = _savitzky_golay_plan(n, window_size, poly_order)
    starts = plan.starts
    lengths = plan.lengths
    weights = plan.weights

    @inbounds for i in 1:n
        start_idx = starts[i]
        len_i = lengths[i]
        acc = 0.0
        @simd for k in 1:len_i
            acc = muladd(weights[i, k], powers[start_idx + k - 1], acc)
        end
        out[i] = acc
    end
    return out
end

function smooth_power_spectrum!(out::AbstractVector{Float64},
                                powers::AbstractVector{Float64};
                                method="savitzky_golay",
                                window_size=11,
                                poly_order=3,
                                sigma=1.0)
    length(out) == length(powers) || error("Output and input lengths must match for smoothing.")

    if method == "none"
        copyto!(out, powers)
    elseif method == "savitzky_golay"
        _savitzky_golay_smooth!(out, powers, window_size, poly_order)
    elseif method == "gaussian"
        _gaussian_smooth!(out, powers, sigma)
    elseif method == "moving_avg"
        _moving_average_smooth!(out, powers, window_size)
    else
        error("Unknown smoothing method: $method")
    end
    return out
end

function smooth_power_spectrum(powers::AbstractVector{Float64};
                               method="savitzky_golay",
                               window_size=11,
                               poly_order=3,
                               sigma=1.0)
    result = similar(powers, Float64)
    smooth_power_spectrum!(result, powers; method=method, window_size=window_size, poly_order=poly_order, sigma=sigma)
    return result
end

function _gaussian_smooth(powers::AbstractVector{Float64}, sigma::Real)
    result = similar(powers, Float64)
    _gaussian_smooth!(result, powers, sigma)
    return result
end

function _savitzky_golay_smooth(powers::AbstractVector{Float64},
                                window_size::Int,
                                poly_order::Int)
    result = similar(powers, Float64)
    _savitzky_golay_smooth!(result, powers, window_size, poly_order)
    return result
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


