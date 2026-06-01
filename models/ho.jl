include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Harmonic oscillator
# Brain source: displacement x, matching the canonical ENEEGMA node definition.
# -----------------------------------------------------------------------------

# Main equations for the harmonic oscillator
function f!(du, u, p, t)
    x1, x2 = u
    ω, ζ, I_ext = p

    du[1] = x2
    du[2] = -ω^2*x1 - 2*ζ*ω*x2 + I_ext(t)
end

# Initial condition
u0 = [1.0, 0.0]

# Simulation settings
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4

solver = Tsit5()
fs = 1.0 / dts

# input settings 
time_vals = 0:dts:tspan[2] |> collect 
input_vals = zeros(length(time_vals)) # zeros(length(time_vals)) # sin.(2*π*5.*time_vals)
I_ext = LinearInterpolation(time_vals, input_vals, extrapolation_bc=Line())

# Parameters
ω = 2π*10.   # Intrinsic frequency
ζ=0.05       # Damping coefficient
p = (ω = ω, ζ=ζ, I_ext=I_ext)

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
model_plot = plot_brain_source_results(brain_df; sampling_rate=fs, model_name="Harmonic Oscillator")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end