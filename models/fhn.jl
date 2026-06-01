include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# FitzHugh-Nagumo oscillator
# Brain source: x1 / v.
# -----------------------------------------------------------------------------

# FitzHugh-Nagumo equations
function f!(du, u, p, t)
    a, b, τ, I = p
    x1, x2 = u

    du[1] = x1 - 1/3*x1^3 - x2 + I
    du[2] = (a + x1 - b*x2) / τ
end


# Parameters
p = (a = 0.7, b = 0.8, τ = 2.5, I = 0.8)

# Initial condition
u0 = [-1.0, 1.0]

# Simulation settings
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = Tsit5()

# Solve the ODE problem
prob = ODEProblem(f!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x1, :x2]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x1)
model_plot = plot_brain_source_results(brain_df; model_name="FitzHugh-Nagumo")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end