include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Wendling neural mass model
# Brain source: excitatory PSP minus slow and fast inhibitory PSPs.
# -----------------------------------------------------------------------------

# Sigmoid function
function sigmoid(V, e0, r, θ)
    return 2 * e0 / (1 + exp(r * (θ - V)))
end

# Main equations
function f!(du, u, p, t)
    A1, a1, A2, a2, A3, a3, A4, a4, C1, C2, C3, C4, C5, C6, C7, e0, r, θ, I_ext = p
    x11, x12, x21, x22, x31, x32, x41, x42, x51, x52 = u

    # pyramidal & its derivative
    du[1] = x12
    du[2] = A1*a1*sigmoid(x21 - x31 - x41, e0, r, θ) - 2*a1*x12 - a1^2*x11

    # excitatory PSP & derivative
    du[3] = x22
    du[4] = A2*a2*(I_ext(t) + C2*sigmoid(C1 * x11, e0, r, θ)) - 2*a2*x22 - a2^2*x21

    # slow‑inhibitory PSP & derivative
    du[5] = x32
    du[6] = A3*a3*(C4 * sigmoid(C3 * x11, e0, r, θ)) - 2*a3*x32 - a3^2*x31

    # fast‑inhibitory PSP & derivative (main)
    du[7] = x42
    du[8] = A4*a4*C7*(sigmoid(C5 * x11 - C6*x51, e0, r, θ)) - 2*a4*x42 - a4^2*x41

    # fast‑inhibitory PSP & derivative (side branch)
    du[9] = x52
    du[10] = A3*a3*(sigmoid(C3 * x11, e0, r, θ)) - 2*a3*x52 - a3^2*x51
end

function g!(du, u, p, t)
    fill!(du, 0.0)
end

# Simulation settings
A1, a1, A2, a2, A3, a3, A4, a4 = 6., 100.0, 6., 100.0, 40., 50.0, 20., 350.
C1, e0, r, θ = 135.0, 2.5, 0.56, 6.0
C2, C3, C4, C5, C6, C7 = 0.8 * C1, 0.25 * C1, 0.25 * C1, 0.3 * C1, 0.1 * C1, 0.8 * C1
μ, σ = 90.0, sqrt(30.) # mean and std for the noise

u0 = 1e-3 .* randn(10)

tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = EM()

# noise settings 1
time_vals = 0:dts:tspan[2] |> collect 
noise_vals = rand(Normal(μ, σ^2), length(time_vals))
I_ext = LinearInterpolation(time_vals, noise_vals, extrapolation_bc=Line())

p = (A1=A1, a1=a1, A2=A2, a2=a2, A3=A3, a3=a3, A4=A4, a4=a4,
     C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, C6=C6, C7=C7,
     e0=e0, r=r, θ=θ, 
     I_ext=I_ext)

# Solve the ODE problem
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x11, :x12, :x21, :x22, :x31, :x32, :x41, :x42, :x51, :x52]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, df -> df.x21 .- df.x31 .- df.x41)
model_plot = plot_brain_source_results(brain_df; model_name="Wendling")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end