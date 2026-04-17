# ============================================================================
# Unified Visualization Functions for Simulation and Optimization
# ============================================================================
# Reusable plotting functions that work for both simulation-only and 
# simulation+observation contexts. Used by save_simulation_results.jl 
# and save_optimization_results.jl to avoid code duplication.

# ============================================================================
# Helper Functions
# ============================================================================

"""
    prepare_psd_for_plotting(powers::Vector{Float64};
                             use_log::Bool=true,
                             data_settings::Union{Nothing, DataSettings}=nothing)::Vector{Float64}

Intelligently prepare PSD powers for plotting by detecting if already log-transformed
and applying log10 only if needed.

# Logic:
1. Check if preprocessing pipeline includes log (from data_settings)
2. Safety check: if any power is negative, it must already be log-transformed
3. Apply log10 only if use_log=true AND powers are not already logged

# Returns:
Powers ready for plotting (in log scale if use_log=true and not already logged)
"""
function prepare_psd_for_plotting(powers::Vector{Float64};
                                  use_log::Bool=true,
                                  data_settings::Union{Nothing, DataSettings}=nothing)::Vector{Float64}
    # Check if powers are already log-transformed by the preprocessing pipeline
    has_log = data_settings !== nothing && hasproperty(data_settings, :psd) ? 
              psd_preproc_has_log(data_settings.psd.preproc_pipeline) : false
    
    # Safety check: if any power values are negative, they must already be log-transformed
    # (raw PSD powers are always positive)
    if !isempty(powers) && any(p -> p < 0, powers)
        has_log = true
    end
    
    # Apply log only if requested AND powers are not already logged
    if use_log && !has_log
        return log10.(max.(powers, eps(Float64)))
    else
        return powers
    end
end

"""
    plot_psd_single(freqs::Vector{Float64}, powers::Vector{Float64};
                    title::String="Power Spectral Density",
                    xlabel::String="Frequency (Hz)",
                    ylabel::String="Log Power",
                    fullfname_fig::Union{Nothing, String}=nothing,
                    use_log::Bool=true,
                    general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing

Plot a single power spectral density curve (simulated only).

# Arguments
- `freqs::Vector{Float64}`: Frequency values
- `powers::Vector{Float64}`: Power values (linear scale)
- `title::String`: Plot title
- `xlabel::String`: X-axis label
- `ylabel::String`: Y-axis label
- `fullfname_fig::Union{Nothing, String}`: Output filename (if nothing, not saved)
- `use_log::Bool`: If true, apply log10 transformation to powers (default: true)
- `general_settings::Union{Nothing, GeneralSettings}`: Settings for checking make_plots flag

# Returns
Nothing (saves to file if fullfname_fig provided)
"""
function plot_psd_single(freqs::Vector{Float64}, powers::Vector{Float64};
                         title::String="Power Spectral Density",
                         xlabel::String="Frequency (Hz)",
                         ylabel::String="Log Power",
                         fullfname_fig::Union{Nothing, String}=nothing,
                         use_log::Bool=true,
                         data_settings::Union{Nothing, DataSettings}=nothing,
                         general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    try
        # Prepare powers for plotting (handles log transformation intelligently)
        plot_powers = prepare_psd_for_plotting(powers; use_log=use_log, data_settings=data_settings)
        p = plot(freqs, plot_powers;
                 label="Simulated",
                 xlabel=xlabel,
                 ylabel=ylabel,
                 title=title,
                 legend=:topright,
                 linewidth=2,
                 size=(800, 400));
        
        if fullfname_fig !== nothing
            savefig(p, fullfname_fig);
        end
    catch e
        vwarn("Failed to plot PSD: $e"; level=2)
    end

    return nothing
end


"""
    plot_psd_comparison(freqs::Vector{Float64}, 
                        simulated_powers::Vector{Float64},
                        observed_powers::Vector{Float64};
                        title::String="Power Spectral Density Comparison",
                        xlabel::String="Frequency (Hz)",
                        ylabel::String="Log Power",
                        fullfname_fig::Union{Nothing, String}=nothing,
                        general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing

Plot power spectral density comparison (simulated vs observed).

# Arguments
- `freqs::Vector{Float64}`: Frequency values
- `simulated_powers::Vector{Float64}`: Simulated power values
- `observed_powers::Vector{Float64}`: Observed power values
- `title::String`: Plot title
- `xlabel::String`: X-axis label
- `ylabel::String`: Y-axis label
- `fullfname_fig::Union{Nothing, String}`: Output filename
- `general_settings::Union{Nothing, GeneralSettings}`: Settings for checking make_plots flag

# Returns
Nothing (saves to file if fullfname_fig provided)
"""
function plot_psd_comparison(freqs::Vector{Float64},
                             simulated_powers::Vector{Float64},
                             observed_powers::Vector{Float64};
                             title::String="Power Spectral Density Comparison",
                             xlabel::String="Frequency (Hz)",
                             ylabel::String="Log Power",
                             fullfname_fig::Union{Nothing, String}=nothing,
                             use_log::Bool=true,
                             data_settings::Union{Nothing, DataSettings}=nothing,
                             general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    # Validate dimensions
    n_freqs = length(freqs)
    n_sim = length(simulated_powers)
    n_obs = length(observed_powers)

    if n_sim != n_obs || n_sim != n_freqs
        vwarn("Cannot plot PSD comparison: dimension mismatch (simulated=$n_sim, observed=$n_obs, freqs=$n_freqs)"; level=2)
        return nothing
    end

    try
        # Prepare powers for plotting (handles log transformation intelligently)
        plot_sim_powers = prepare_psd_for_plotting(simulated_powers; use_log=use_log, data_settings=data_settings)
        plot_obs_powers = prepare_psd_for_plotting(observed_powers; use_log=use_log, data_settings=data_settings)
        
        p = plot(freqs, plot_sim_powers;
                 label="Simulated",
                 xlabel=xlabel,
                 ylabel=ylabel,
                 title=title,
                 legend=:topright,
                 linewidth=2,
                 size=(800, 400));
        
        plot!(p, freqs, plot_obs_powers;
              label="Observed",
              linewidth=2,
              color=:black,
              alpha=0.8);
        
        if fullfname_fig !== nothing
            savefig(p, fullfname_fig);
        end
    catch e
        vwarn("Failed to plot PSD comparison: $e"; level=2)
    end

    return nothing
end


"""
    plot_timeseries_windows(times::Vector{Float64}, 
                            simulated_signal::Vector{Float64};
                            times::Union{Nothing, Vector{Float64}}=nothing,
                            observed_signal::Union{Nothing, Vector{Float64}}=nothing,
                            zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                            title_prefix::String="",
                            fullfname_fig::Union{Nothing, String}=nothing,
                            general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing

Plot timeseries with full and zoomed windows. Supports both simulation-only and 
simulated+observed comparison contexts.

For simulation-only: Shows simulated signal in both windows.
For comparison: Shows both simulated (blue) and observed (black) in each window.

# Arguments
- `times::Vector{Float64}`: Time points for simulated signal
- `simulated_signal::Vector{Float64}`: Simulated signal values
- `times::Union{Nothing, Vector{Float64}}`: Time points for observed signal (optional)
- `observed_signal::Union{Nothing, Vector{Float64}}`: Observed signal values (optional)
- `zoom_window::Tuple{Float64, Float64}`: Time range for zoomed view (in seconds)
- `title_prefix::String`: Prefix for plot titles
- `fullfname_fig::Union{Nothing, String}`: Output filename
- `general_settings::Union{Nothing, GeneralSettings}`: Settings for checking make_plots flag

# Returns
Nothing (saves to file if fullfname_fig provided)
"""
function plot_timeseries_windows(times::Vector{Float64},
                                 simulated_signal::Vector{Float64};
                                 observed_signal::Union{Nothing, Vector{Float64}}=nothing,
                                 zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                                 title_prefix::String="",
                                 fullfname_fig::Union{Nothing, String}=nothing,
                                 general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    has_observed = observed_signal !== nothing

    try
        p = plot(layout=(2, 1), size=(900, 500), legend=:topright);

        # Full timeseries window
        plot!(p[1], times, simulated_signal; label="Simulated", xlabel="", ylabel="");
        if has_observed
            plot!(p[1], times, observed_signal; label="Observed", color=:black);
        end

        # Zoomed timeseries window - check if zoom_window is within signal range
        time_min = minimum(times)
        time_max = maximum(times)
        safe_zoom_start = max(zoom_window[1], time_min)
        safe_zoom_end = min(zoom_window[2], time_max)
        safe_zoom = (safe_zoom_start, safe_zoom_end)
        
        plot!(p[2], times, simulated_signal; label="Simulated", xlabel="Time (s)", ylabel="", xlims=safe_zoom);
        if has_observed
            obs_time_min = minimum(times)
            obs_time_max = maximum(times)
            obs_safe_zoom_start = max(zoom_window[1], obs_time_min)
            obs_safe_zoom_end = min(zoom_window[2], obs_time_max)
            obs_safe_zoom = (obs_safe_zoom_start, obs_safe_zoom_end)
            plot!(p[2], times, observed_signal; label="Observed", color=:black, xlims=obs_safe_zoom);
        end

        if title_prefix != ""
            plot!(p, plot_title=title_prefix);
        end

        if fullfname_fig !== nothing
            savefig(p, fullfname_fig);
        end
    catch e
        vwarn("Failed to plot timeseries windows: $e"; level=2)
    end

    return nothing
end


"""
    plot_simulation_results(times::Vector{Float64},
                            signal::Vector{Float64},
                            freqs::Vector{Float64},
                            powers::Vector{Float64};
                            zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                            signal_name::String="Signal",
                            fullfname_fig::Union{Nothing, String}=nothing,
                            use_log::Bool=true,
                            general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing

Create a composite 3-panel plot for simulation results:
  - Subplot 1: Full timeseries
  - Subplot 2: Zoomed timeseries (2-second window) with ylims based on window segment
  - Subplot 3: Power spectral density (PSD) - log scale by default

Used for visualization of simulation-only runs.

# Arguments
- `times::Vector{Float64}`: Time points for signal
- `signal::Vector{Float64}`: Signal values
- `freqs::Vector{Float64}`: Frequency values for PSD
- `powers::Vector{Float64}`: Power values for PSD (linear scale)
- `zoom_window::Tuple{Float64, Float64}`: Time range for zoomed view
- `signal_name::String`: Name of signal for plot labels
- `fullfname_fig::Union{Nothing, String}`: Output filename
- `use_log::Bool`: If true, apply log10 transformation to PSD (default: true)
- `general_settings::Union{Nothing, GeneralSettings}`: Settings for checking make_plots flag

# Returns
Nothing (saves to file if fullfname_fig provided)
"""
function plot_simulation_results(times::Vector{Float64},
                                 signal::Vector{Float64},
                                 freqs::Vector{Float64},
                                 powers::Vector{Float64};
                                 zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                                 signal_name::String="Signal",
                                 fullfname_fig::Union{Nothing, String}=nothing,
                                 use_log::Bool=true,
                                 data_settings::Union{Nothing, DataSettings}=nothing,
                                 general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    try
        # Create 3-panel subplot layout
        p = plot(layout=(3, 1), size=(900, 900), legend=:topright);

        # Subplot 1: Full timeseries
        plot!(p[1], times, signal;
              label=signal_name,
              xlabel="",
              ylabel="Amplitude",
              title="Full Timeseries",
              linewidth=1.5);

        # Subplot 2: Zoomed timeseries (2-second window)
        # Calculate ylims based on the signal segment within the zoom window
        zoom_indices = findall(t -> zoom_window[1] <= t <= zoom_window[2], times)
        if !isempty(zoom_indices)
            zoom_signal = signal[zoom_indices]
            zoom_min = minimum(zoom_signal)
            zoom_max = maximum(zoom_signal)
            # Add 10% margin to zoom limits
            margin = 0.1 * (zoom_max - zoom_min)
            zoom_ylims = (zoom_min - margin, zoom_max + margin)
        else
            zoom_ylims = :auto
        end
        
        plot!(p[2], times, signal;
              label=signal_name,
              xlabel="Time (s)",
              ylabel="Amplitude",
              title="Zoomed Window (2 sec)",
              xlims=zoom_window,
              ylims=zoom_ylims,
              linewidth=1.5);

        # Subplot 3: PSD (log scale)
        # Prepare powers for plotting (handles log transformation intelligently)
        plot_powers = prepare_psd_for_plotting(powers; use_log=use_log, data_settings=data_settings)
        plot!(p[3], freqs, plot_powers;
              label="PSD",
              xlabel="Frequency (Hz)",
              ylabel="Log Power",
              title="Power Spectral Density",
              legend=:topright,
              linewidth=2);

        if fullfname_fig !== nothing
            savefig(p, fullfname_fig);
        end
    catch e
        vwarn("Failed to create simulation results plot: $e"; level=2)
    end

    return nothing
end
