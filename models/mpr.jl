include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))


# ------------------------------------------------------------------
# Montbrió–Pazó–Roxin (2015) population model
#   ẊR = Δ/π + 2 R V
#   ẊV = V^2 + η₀ + I(t) - (π R)^2
# https://docs.thevirtualbrain.org/_modules/tvb/rateML/generatedModels/montbrio.html
# Brain source: population firing rate R, matching the canonical ENEEGMA node definition.
# ------------------------------------------------------------------

# Main equations (allow I to be a number or a function I(t))
#= function f!(du, u, p, t)
    τ, J, Δ, η0, I = p.τ, p.J, p.Δ, p.η0, p.I
    R, V = u

    It = I isa Number ? I : I(t)
    du[1] = (1/τ) * (Δ/(π*τ) + 2R*V)
    du[2] = (1/τ) * (V^2 + η0 + It - (τ*π*R)^2 + J*τ*R)
end
  =#
function f!(du, u, p, t)
    τ, J, Δ, η0, I = p.τ, p.J, p.Δ, p.η0, p.I
    x1, x2 = u

    It = I isa Number ? I : I(t)
    du[1] = Δ/(π*τ^2) + (2/τ)*x1*x2
    du[2] = (η0+3)/τ + J*x1 - (τ*π^2)*x1^2 + (1/τ)*x2^2 + (1/τ)*It
end


# ----------------------------
# Parameters and input
# ----------------------------
# Example: constant input
# Example (optional): time-dependent drive (step at t=5 s)
# I_fun = t -> (t < 5.0 ? 0.0 : 0.5)

p = (τ = 0.1,
    J = 15.5,
    Δ = 1.,          # half-width of η distribution
    η0 = -4.6        # mean excitability
)

# ----------------------------
# Initial condition & sim setup
# ----------------------------
u0 = [0.5, 0.0]        # [R(0), V(0)]
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = Tsit5()

# ----------------------------
# Input
# ----------------------------

fs  = 1/dts
tgrid = collect(0.0:dts:tspan[2])

Random.seed!(1)
I0 = 0.0
σI = 1.               # tune
#Ivec = I0 .+ σI .* randn(length(tgrid))
Ivec = rand(Normal(I0, σI), length(tgrid))
#Ivec = 3 .+ rand(Normal(0., 1.), length(tgrid))

I_interp = LinearInterpolation(tgrid, Ivec, extrapolation_bc=Flat())

# add I_interp to parameter tuple p 
p = merge(p, (I = I_interp,))

# ----------------------------
# Solve ODE
# ----------------------------
prob = ODEProblem(f!, u0, tspan, p)
sol  = solve(prob, solver, dt=dt, saveat=dts)

# ----------------------------
# Post-processing & plots
# ----------------------------
signal_cols = [:R, :V]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)

brain_df = brain_source_dataframe(sol_df, :R)
model_plot = plot_brain_source_results(brain_df; model_name="Montbrio-Pazo-Roxin")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end