# Example 2: Network Simulation
# ==============================
# This example demonstrates how to:
#   1. Set up a network 
#   2. Build the network structure
#   3. Run simulations
#   4. Access and visualize results as DataFrame
#   5. Resample initial conditions and simulate again

using ENEEGMA
using Plots
using DataFrames

println("\nENEEGMA Network Simulation Example\n")

# ============================================================================
# SETTINGS
# ============================================================================
settings = create_default_settings();
settings.simulation_settings.tspan = (0.0, 100.0)  # Simulate for 100 seconds
# print_settings_summary(settings; section="general_settings");
# print_settings_summary(settings; section="network_settings");
# print_settings_summary(settings; section="simulation_settings");

# ============================================================================
# BUILD NETWORK
# ============================================================================
net = build_network(settings);
# Display network parameters and variable (state) initial condition ranges
# print_params_summary(net.params);
# print_vars_summary(net.vars);

# ============================================================================
# SIMULATION - Simulate network and get DataFrame with named columns
# ============================================================================
df = simulate_network(net)

# Plot first or EEG signal over subset of time points
fs = Int(1/settings.simulation_settings.saveat)
x_range = 1*fs:min(size(df, 1), 2*fs)
signal_name = get_eeg_signal(settings, df)
signal = df[!, Symbol(signal_name)]
ph = plot(df.time[x_range], signal[x_range], 
          xlabel="Time (s)", ylabel="Signal", 
          title="Simulated Signal: $signal_name")
display(ph)

freqs, powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs; xlims=(1., 50.))
plot(freqs, powers, xlabel="Frequency (Hz)", ylabel="Power", title="Welch PSD of Simulated Signal")

# ============================================================================
# Resample initial conditions with different seed and simulate again
# ============================================================================
println("\n=== Resampling Initial Conditions and Simulating Again ===")
new_inits = sample_inits(net.vars; seed=192)
df2 = simulate_network(net; new_inits=new_inits)

ph2 = plot(df2.time[x_range], df2[x_range, Symbol(signal_name)], 
           xlabel="Time (ms)", ylabel="Signal", 
           title="Simulated Signal (Resampled Inits): $signal_name")
display(ph2)

# ============================================================================
# Change parameters and simulate again
# ============================================================================
println("\n=== Modifying Parameters and Simulating Again ===")
new_param_values = Dict("N1₊tscale1" => 1.0, "N1₊c18" => 10.)
update_param_defaults!(net.params, new_param_values)
df3 = simulate_network(net, new_params=new_param_values)
ph3 = plot(df3.time[x_range], df3[x_range, Symbol(signal_name)], 
           xlabel="Time (ms)", ylabel="Signal", 
           title="Simulated Signal (Modified Params): $signal_name")
display(ph3)

# PSD
fs = 1.0 / settings.simulation_settings.saveat
signal = df3[!, Symbol(signal_name)]
plot(signal[1000:end], xlabel="Time Points", ylabel="Signal", title="Simulated Signal (First 1000 Time Points)")
model_freqs, modeled_powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs)
plot(model_freqs, modeled_powers, xlabel="Frequency (Hz)", ylabel="Power", title="PSD of Simulated Signal", yscale=:log10)

# ============================================================================
# Change parameters and simulate again 2
# ============================================================================
println("\n=== Modifying Parameters and Simulating Again ===")
new_param_values = Dict("N1₊tscale1" => 1.777, 
                        "N1₊c11" => 37.657,
                        "N1₊c12" => -3.265,
                        "N1₊c13" => 33.528,
                        "N1₊c14" => 4.06,
                        "N1₊c15" => -0.293,
                        "N1₊c16" => 18.658,
                        "N1₊c17" => 2.864,
                        "N1₊c18" => 7.9)
update_param_defaults!(net.params, new_param_values)
df4 = simulate_network(net, new_params=new_param_values)
x_range = 1*fs:min(size(df4, 1), 10*fs)
ph4 = plot(df4.time[x_range], df4[x_range, Symbol(signal_name)], 
           xlabel="Time (ms)", ylabel="Signal", 
           title="Simulated Signal (Modified Params): $signal_name")
display(ph4)
signal = df4[!, Symbol(signal_name)]
model_freqs, modeled_powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs)
plot(model_freqs, modeled_powers, xlabel="Frequency (Hz)", ylabel="Power", 
     title="PSD of Simulated Signal")

     