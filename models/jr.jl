include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Jansen-Rit cortical column
# Brain source: excitatory PSP minus inhibitory PSP (x21 - x31).
# -----------------------------------------------------------------------------

# Sigmoid function
function sigmoid(V, e0, r, θ)
    return 2 * e0 / (1 + exp(r * (θ - V)))
end

# Main equations
function f!(du, u, p, t)
    A0, a0, A1, a1, A2, a2, C1, C2, C3, C4, e0, r, θ, I_ext = p

    x11, x12, x21, x22, x31, x32 = u
    
    du[1] = x12
    du[2] = A0*a0*sigmoid(x21 - x31, e0, r, θ) - 2*a0*x12 - a0^2*x11
    du[3] = x22
    du[4] = A1*a1*(I_ext(t) + C2*sigmoid(C1 * x11, e0, r, θ)) - 2*a1*x22 - a1^2*x21
    du[5] = x32
    du[6] = A2*a2*(C4 * sigmoid(C3 * x11, e0, r, θ)) - 2*a2*x32 - a2^2*x31
end

function g!(du, u, p, t)
    fill!(du, 0.0)
end

# Simulation settings
A0, a0, A1, a1, A2, a2 = 3.25, 100.0, 3.25, 100.0, 22., 50.0
C1, e0, r, θ = 135.0, 2.5, 0.56, 6.0
C2, C3, C4 = 0.8 * C1, 0.25 * C1, 0.25 * C1

u0 = [0.39057, -204.28131, -3.88133, -0.11387, -7.68439, -253.32597]

tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = EM()

# noise settings 1
time_vals = 0:dts:tspan[2] |> collect 
noise_vals = rand(length(time_vals)) .* 320
I_ext = LinearInterpolation(time_vals, noise_vals, extrapolation_bc=Line())

p = (A0=A0, a0=a0, A1=A1, a1=a1, A2=A2, a2=a2,
     C1=C1, C2=C2, C3=C3, C4=C4, 
     e0=e0, r=r, θ=θ, 
     I_ext=I_ext)

# Solve the ODE problem
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x11, :x12, :x21, :x22, :x31, :x32]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, df -> df.x21 .- df.x31)
model_plot = plot_brain_source_results(brain_df; model_name="Jansen-Rit")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end