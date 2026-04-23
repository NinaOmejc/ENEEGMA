# Example 2: Network Simulation
# ============================================================================
# This example demonstrates how to:
#   1. Set up a network 
#   2. Build the network structure
#   3. Run simulations
#   4. Save and visualize results as DataFrame
#   5. Resample initial conditions or change parameters and simulate again

using ENEEGMA
using Plots
using DataFrames

println("\nENEEGMA Network Simulation Example\n")

# ============================================================================
# Step 1: Settings
# ============================================================================
settings = create_default_settings();
# print_settings_summary(settings; section="general_settings");
# print_settings_summary(settings; section="network_settings");
# print_settings_summary(settings; section="simulation_settings");

# ============================================================================
# Step 2: Build Network
# ============================================================================
net = build_network(settings);
# Display network parameters and variable (state) initial condition ranges
# print_params_summary(net.params);
# print_vars_summary(net.vars);

# ============================================================================
# Step 3: Simulate Network and Save Results
# ============================================================================
df = simulate_network(net)

# Save simulation results: creates simulation_N/ folder with composite plot and CSV
# The composite plot includes: full timeseries, zoomed 2-sec window, and PSD
save_simulation_results(df, net, settings)

# ============================================================================
# Manual Plotting Example: Using plot_simulation_results directly
# ============================================================================
# You can also use plot_simulation_results directly for custom visualization
fs = 1.0 / settings.simulation_settings.saveat
df_sources = ENEEGMA.extract_brain_sources(settings, net, df)

psd_dict = ENEEGMA.compute_psd_for_all_sources(df_sources, fs)

# Build output file names using standardized format: exp_name_net_name_simulated
# Save composite plot (3xN panels) for all source signals
path_plot = "$(settings.general_settings.exp_name)_$(net.name)_simulated.png"
ENEEGMA.plot_simulation_results(df_sources; psd_dict=psd_dict,
                                zoom_window=(2.0, 5.0),
                                fullfname_fig=path_plot,
                                data_settings=settings.data_settings,
                                general_settings=settings.general_settings)

# ============================================================================
# Step 4: Resample Initial Conditions and Simulate Again
# ============================================================================
println("\nPART 2: Resampling Initial Conditions and Simulating Again")
new_inits = sample_inits(net.vars; seed=192)
df2 = simulate_network(net; new_inits=new_inits)
println("Simulation completed with resampled initial conditions")

# ============================================================================
# Step 5: Change Parameters and Simulate Again
# ============================================================================
println("\nPART 3: Modifying Parameters and Simulating Again")
new_param_values = Dict("N1₊c11" => 37.657,
                        "N1₊c12" => -3.265,
                        "N1₊c13" => 33.528,
                        "N1₊c14" => 4.06,
                        "N1₊c15" => -0.293,
                        "N1₊c16" => 18.658,
                        "N1₊c17" => 2.864,
                        "N1₊c18" => 7.9)
update_param_defaults!(net, new_param_values);
df4 = simulate_network(net);
save_simulation_results(df4, net, settings)
println("Simulation completed with further parameter modifications")
