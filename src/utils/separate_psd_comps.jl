using ENEEGMA
using Plots
using DataFrames
using Statistics

"""
    separate_psd_components(freqs, powers, ls;
        fmin_fit=ls.fmin, fmax_fit=ls.fmax,
        peak_windows=nothing, min_points=5)

Split a PSD into an aperiodic 1/f background and a broad peak component.
The model is linear in log-log space: log10(PSD) = a + b * log10(f).

Returns a NamedTuple with log-space and power-space components plus fit metadata.
Best results come from PSDs that are log-transformed but not normalized.
"""
function separate_psd_components(freqs::AbstractVector{<:Real},
                                 powers::AbstractVector{<:Real},
                                 ls::LossSettings;
                                 fmin_fit::Float64 = ls.fmin,
                                 fmax_fit::Float64 = ls.fmax,
                                 background_quantile::Union{Nothing, Real} = 0.5,
                                 background_window_hz::Float64 = 6.0,
                                 peak_windows = nothing,
                                 min_points::Int = 5)
    length(freqs) == length(powers) || error("freqs and powers must have the same length.")
    n = length(freqs)
    n == 0 && error("Empty PSD.")

    f = Float64.(freqs)
    logpsd = ENEEGMA.psd_preproc_has_log(ls.psd_preproc) ?
        Float64.(powers) :
        log10.(max.(Float64.(powers), eps(Float64)))

    if peak_windows === nothing
        peak_windows = ENEEGMA.detect_peak_windows(f, logpsd, ls)
    end
    peak_mask = ENEEGMA._mask_from_windows(f, peak_windows)

    fit_mask = (f .>= fmin_fit) .& (f .<= fmax_fit) .& .!peak_mask .& (f .> 0)
    if count(fit_mask) < min_points
        fit_mask = (f .> 0) .& .!peak_mask
    end
    if count(fit_mask) < min_points
        error("Not enough points to fit background. Adjust fmin/fmax or peak settings.")
    end

    x = log10.(f[fit_mask])
    if background_quantile === nothing
        y = logpsd[fit_mask]
    else
        df = length(f) > 1 ? median(diff(f)) : NaN
        bins = _suggest_window_bins_local(background_window_hz, df, 11; min_bins=7)
        baseline = _running_quantile_local(logpsd, bins, background_quantile)
        y = baseline[fit_mask]
    end
    a, b = _linear_fit(x, y)

    log_background = similar(logpsd)
    pos_mask = f .> 0
    log_background[pos_mask] = a .+ b .* log10.(f[pos_mask])
    if any(.!pos_mask)
        first_val = log_background[findfirst(pos_mask)]
        log_background[.!pos_mask] .= first_val
    end

    log_peak = logpsd .- log_background
    background_power = 10.0 .^ log_background
    peak_power = max.(10.0 .^ logpsd .- background_power, 0.0)

    return (freqs=f,
            logpsd=logpsd,
            log_background=log_background,
            log_peak=log_peak,
            background_power=background_power,
            peak_power=peak_power,
            peak_windows=peak_windows,
            peak_mask=peak_mask,
            fit=(intercept=a, slope=b),
            fit_mask=fit_mask)
end

function _running_quantile_local(x::AbstractVector{<:Real}, w::Int, q::Real)
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

function _suggest_window_bins_local(width_hz::Float64, df::Float64, fallback::Int; min_bins::Int=3)
    if isnan(df) || df <= 0
        bins = fallback
    else
        bins = max(min_bins, Int(round(width_hz / df)))
    end
    bins = max(1, bins)
    return isodd(bins) ? bins : bins + 1
end

function _linear_fit(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || error("x and y must have the same length.")
    n == 0 && return (0.0, 0.0)
    mx = mean(x)
    my = mean(y)
    num = 0.0
    den = 0.0
    @inbounds for i in 1:n
        dx = float(x[i]) - mx
        dy = float(y[i]) - my
        num += dx * dy
        den += dx * dx
    end
    slope = den == 0 ? 0.0 : num / den
    intercept = my - slope * mx
    return intercept, slope
end
