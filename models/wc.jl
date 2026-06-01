include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Wilson-Cowan excitatory/inhibitory neural mass
# Brain source: excitatory activity x11, matching the canonical ENEEGMA node definition.
# -----------------------------------------------------------------------------

# Sigmoid function
function sigmoid(V, a, θ)
    return 1 / (1 + exp(-a*(V - θ))) - (1 / (1 + exp(a*θ)))
end

# Main equations 
function f!(du, u, p, t)
    w_EE, w_EI, w_IE, w_II, τ_E, τ_I, a_E, a_I, θ_E, θ_I, r_E, r_I, I_E, I_I = p
    x11, x12 = u[1], u[2]

    du[1] = (-x11 + (1 - r_E*x11)*sigmoid(w_EE * x11 - w_EI * x12 + I_E, a_E, θ_E)) / τ_E
    du[2] = (-x12 + (1 - r_I*x12)*sigmoid(w_IE * x11 - w_II * x12 + I_I, a_I, θ_I)) / τ_I
end

# Parameters for the Jansen-Rit model
p = (w_EE=16.0, w_EI=26.0, w_IE=20.0, w_II=1.0,
     τ_E=4e-2, τ_I=4e-2,
     a_E=1.0, a_I=1.0,
     θ_E=5., θ_I=20.,
     r_E=0.0, r_I=0.0,
     I_E=5., I_I=5.)

# Simulation settings
u0 = [0.949, 0.12] # rand(2)

tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = Tsit5()

# Solve the ODE problem
prob = ODEProblem(f!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)
signal_cols = [:x11, :x12]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x11)
model_plot = plot_brain_source_results(brain_df; model_name="Wilson-Cowan")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end