# Example 2: Grammar Sampling and Network Simulation
# ====================================================
# This example demonstrates how to:
#   1. Set up a network with grammar-sampled node models
#   2. Build the network structure
#   3. Run simulations with multiple random initializations
#   4. Output settings and results
using ENEEGMA
using Plots

settings = create_default_settings();

print_settings_summary(settings; section="network_settings");
print_settings_summary(settings; section="simulation_settings");

# Build Network from Settings
net = build_network(settings);

# Display network parameters
print_params_summary(net.params);

# Display variable (state) initial condition ranges
print_vars_summary(net.vars);

simulations = simulate_network(net);    

# Plot time series
eeg_output = "N1₊x11"
x_range = 1000:2000
times = simulations[1].time[x_range]
signal = simulations[1][x_range, eeg_output]
ph = plot(times, signal, xlabel="Time (s)", ylabel="Signal", title="Simulated Signal")
display(ph)

