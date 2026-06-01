using DataFrames
using DifferentialEquations
using Distributions
using Interpolations
using Plots
using Random
using Statistics
using ENEEGMA

function brain_source_dataframe(sol_df::DataFrame, source; source_name::Symbol=:brain_source)
    brain_df = DataFrame(time=sol_df.time)
    brain_df[!, source_name] = source isa Symbol ? sol_df[!, source] : source(sol_df)
    return brain_df
end

function infer_sampling_rate(df::DataFrame)
    times = collect(skipmissing(df.time))
    length(times) < 2 && error("Cannot infer sampling rate from fewer than two time points.")
    dt = median(diff(times))
    dt <= 0 && error("Time values must be strictly increasing.")
    return 1.0 / dt
end

function compute_model_psd_for_all_sources(df_sources::DataFrame, 
                                           fs::Real;
                                           data_settings::Union{Nothing, DataSettings}=nothing,
                                           loss_settings::Union{Nothing, LossSettings}=nothing)
    # Use the unified PSD computation function from ENEEGMA
    return ENEEGMA.compute_psd_for_all_sources(df_sources, fs;
                                               data_settings=data_settings,
                                               loss_settings=loss_settings)
end

function plot_brain_source_results(df_sources::DataFrame;
                                   model_name::AbstractString="Model",
                                   zoom_window::Tuple{<:Real,<:Real}=(2.0, 5.0),
                                   sampling_rate::Union{Nothing, Real}=nothing)
    fs = sampling_rate === nothing ? infer_sampling_rate(df_sources) : sampling_rate
    psd_dict = compute_model_psd_for_all_sources(df_sources, fs)

    source_cols = [col for col in names(df_sources) if String(col) != "time"]
    isempty(source_cols) && error("No brain-source columns found.")

    n_sources = length(source_cols)
    p = plot(layout=(2, 2*n_sources),
             size=(1000 * n_sources, 800),
             legend=:topright,
             titlefontsize=11,
             guidefontsize=9,
             tickfontsize=8)

    subplot_index(row_idx, col_idx) = (row_idx - 1) * (2*n_sources) + col_idx

    for (col_idx, col) in enumerate(source_cols)
        source_name = String(col)
        times = df_sources.time
        signal = df_sources[!, col]
        freqs, powers = get(psd_dict, source_name, (Float64[], Float64[]))
        freqs_bounded = freqs[freqs .<= 100.0]
        powers_bounded = powers[1:length(freqs_bounded)]

        # Row 1: Full timeseries
        plot!(p[subplot_index(1, 2*col_idx - 1)], times, signal;
              label=source_name,
              xlabel="",
              ylabel="Amplitude",
              linewidth=1.2,
              title=col_idx == 1 ? "$(model_name): full timeseries" : "Full timeseries")

        # Row 1: Zoomed window
        zoom_idx = findall(t -> zoom_window[1] <= t <= zoom_window[2], times)
        ylims = :auto
        if !isempty(zoom_idx)
            zoom_signal = signal[zoom_idx]
            y_min, y_max = extrema(zoom_signal)
            margin = max(0.1 * (y_max - y_min), eps(Float64))
            ylims = (y_min - margin, y_max + margin)
        end
        plot!(p[subplot_index(1, 2*col_idx)], times, signal;
              label=source_name,
              xlabel="Time (s)",
              ylabel="Amplitude",
              xlims=zoom_window,
              ylims=ylims,
              linewidth=1.2,
              title="Zoomed window")

        # Row 2: Linear spectrum
        plot!(p[subplot_index(2, 2*col_idx - 1)], freqs_bounded, powers_bounded;
              label="PSD",
              xlabel="Frequency (Hz)",
              ylabel="Power",
              linewidth=1.5,
              title="PSD (linear)")

        # Row 2: Log spectrum
        plot_powers_log = log10.(max.(powers_bounded, eps(Float64)))
        plot!(p[subplot_index(2, 2*col_idx)], freqs_bounded, plot_powers_log;
              label="PSD",
              xlabel="Frequency (Hz)",
              ylabel="Log power",
              linewidth=1.5,
              title="PSD (log)")
    end

    display(p)

    return p
end
