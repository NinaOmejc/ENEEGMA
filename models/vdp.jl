include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Van der Pol oscillator
# Brain source: oscillator displacement x, matching the canonical ENEEGMA node definition.
# -----------------------------------------------------------------------------

function f!(du, u, p, t)
    x, y = u
    du[1] = y
    du[2] = p.mu * (1 - x^2) * y - x + p.I_ext(t)
end

u0 = [0.0, 1.0]
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = Tsit5()

time_vals = collect(0.0:dts:tspan[2])
input_vals = zeros(length(time_vals))
I_ext = LinearInterpolation(time_vals, input_vals, extrapolation_bc=Line())

p = (mu = 1.0, I_ext = I_ext)

prob = ODEProblem(f!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x, :y]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x)
model_plot = plot_brain_source_results(brain_df; model_name="Van der Pol")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end