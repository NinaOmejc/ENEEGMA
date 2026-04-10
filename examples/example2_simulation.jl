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
x_range = 200:min(size(df, 1), 2000)
signal_name = get_eeg_signal(settings, df)
ph = plot(df.time[x_range], df[x_range, Symbol(signal_name)], 
          xlabel="Time (ms)", ylabel="Signal", 
          title="Simulated Signal: $signal_name")
display(ph)

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
