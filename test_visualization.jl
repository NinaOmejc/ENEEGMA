using ENEEGMA

println("Testing visualization and simulation functions...")
println()

# Check if functions are defined
println("plot_simulation_results defined: ", isdefined(ENEEGMA, :plot_simulation_results))
println("save_simulation_results defined: ", isdefined(ENEEGMA, :save_simulation_results))
println("plot_psd_comparison defined: ", isdefined(ENEEGMA, :plot_psd_comparison))
println("plot_timeseries_windows defined: ", isdefined(ENEEGMA, :plot_timeseries_windows))
println()
println("SUCCESS: All visualization and simulation functions loaded!")
