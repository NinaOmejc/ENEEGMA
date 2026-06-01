using StochasticDiffEq

include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Moran-David-Friston neural mass model
# Brain source: primary-population net potential used by this standalone form.
# -----------------------------------------------------------------------------

# MDF Sigmoid function
function sigmoid(V, ρ₁, ρ₂)
    return (1 / (1 + exp(-ρ₁ * (V - ρ₂)))) - (1 / (1 + exp(ρ₁ * ρ₂)))
end

# Main equations for MDF model
function f!(du, u, p, t)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = u
    He, Hi, κe, κi, γ₁, γ₂, γ₃, γ₄, γ₅, ρ₁, ρ₂, a, I = p

    # Dynamics

    # Recurrent Excitatory Loop (self-excitation loop) (From Excitatory Interneurons to PYR):
    du[1] = x2
    du[2] = κe * He * γ₂ * sigmoid(x6, ρ₁, ρ₂) - 2 * κe * x2 - κe^2 * x1

    # Inhibitory Population Dynamics (Inhibitory Interneurons)
    du[3] = x4
    du[4] = κi * Hi * γ₄ * sigmoid(x12, ρ₁, ρ₂) - 2 * κi * x4 - κi^2 * x3

    # x7 = net potential for primary population (	Drives output from primary cells)
    du[5] = x2 - x4    

    # Excitatory Interneurons Dynamics
    du[6] = x7
    du[7] = κe * He * (γ₁ * sigmoid(x5 - a, ρ₁, ρ₂) + I(t)) - 2 * κe * x7 - κe^2 * x6

    # Inhibitory Input (from PYR to Inhibitory Interneurons):
    du[8] = x9
    du[9] = κe * He * γ₃ * sigmoid(x5, ρ₁, ρ₂) - 2 * κe * x9 - κe^2 * x8

    # Recurrent Inhibitory Input (from IIN to IIN):
    du[10] = x11
    du[11] = κi * Hi * γ₅ * sigmoid(x12, ρ₁, ρ₂) - 2 * κi * x11 - κi^2 * x10

    # x12 = net potential for downstream populations
    du[12] = x9 - x11

    return du
    
end

function g!(du, u, p, t)
    du .= 0.0
    # du[2] = sqrt(1e-4)
end


# Initial conditions
u0 = zeros(12)

# Simulation settings
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = EM() # SRIW1 is a strong order 1 method suitable for SDEs

# input
time_vals = 0:dts:tspan[2] |> collect 
noise_vals = rand(Normal(0., 25.), length(time_vals))
I_ext = LinearInterpolation(time_vals, noise_vals, extrapolation_bc=Line())
# trange = 1000:30_000
# plot(time_vals[trange], noise_vals[trange], label="I_ext", xlabel="Time (s)", ylabel="Amplitude", title="External Input Signal", legend=:topright)
# histogram(noise_vals, bins=100, title="Histogram of External Input Signal", xlabel="Amplitude", ylabel="Frequency", ylimits=(0, 8.5*10^4))
# histogram(noise_vals, bins=100, title="Histogram of External Input Signal", xlabel="Amplitude", ylabel="Frequency")

# Parameters
p = (
    He = 10.0,
    Hi = 22.0,
    κe = 250.0,
    κi = 62.5,
    γ₁ = 128.0,
    γ₂ = 128.0,
    γ₃ = 64.0,
    γ₄ = 64.0,
    γ₅ = 1.0,
    ρ₁ = 2.0,
    ρ₂ = 1.0,
    a = 0.0,
    I_ext
)

# Solve the SDE problem
prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10, :x11, :x12]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x5)
model_plot = plot_brain_source_results(brain_df; model_name="Moran-David-Friston")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end