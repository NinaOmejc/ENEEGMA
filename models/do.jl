include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# Duffing oscillator equations
function f!(du, u, p, t)
    δ, α, β, γ, f = p
    x1, x2 = u

    du[1] = x2
    du[2] = -δ*x2 - α*x1 - β*x1^3 + γ*cos(2*π*f*t)
end

# Parameters
# Nonchaotic Duffing oscillator parameters
p = (δ = 0.3, α = -1.0, β = 1.0, γ = 0.2, f = 1.2)
# p = (δ = -0.03, α = 1.0, β = -1.0, γ = 0.5, f = 5.0)
# p = (δ = -0.05, α = 100.0, β = -200.0, γ = 1.0, f = 10.0)

# Chaotic Duffing oscillator parameters
# p = (δ = -0.2, α = 1.0, β = -1.0, γ = 0.3, f = 10.)

# Initial condition
u0 = [0.0, 1.0]

# Simulation settings
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = Tsit5()

# Solve the ODE problem
prob = ODEProblem(f!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x, :v]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x)
model_plot = plot_brain_source_results(brain_df; model_name="Duffing Oscillator")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end