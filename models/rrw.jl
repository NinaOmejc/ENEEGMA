include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Robinson-Rennie-Wright corticothalamic model
# Brain source: excitatory pulse density Phi_e.
# -----------------------------------------------------------------------------

# Sigmoid functions
function sigmoid(V; Qmax, θ, σ)
    return Qmax / (1 + exp(-(V - θ)/σ))
end

function f!(du, u, h, p, t)
    α, β, γ, τ, Qmax, θ, σ, Qmax_r, σ_r, Φn0, Φn,
        nu_ee, nu_ei, nu_es, nu_ie, nu_ii, nu_is,
        nu_se, nu_sr, nu_sn, nu_re, nu_rs, S = p

    Ve, Vex, Φe, Φex, Vi, Vix, Vs, Vsx, Vr, Vrx = u
    hhist = h(p, t - τ)
    hΦe = hhist[3]
    hVs = hhist[7]
    
    Is = nu_se*hΦe + nu_sr*sigmoid(Vr; Qmax=Qmax_r, θ=θ, σ=σ_r) + S(t) 
    du[7] = Vsx
    du[8] = -(α + β)*Vsx + α*β*(-Vs + Is)
    
    Ir = nu_re*hΦe + nu_rs*sigmoid(Vs; Qmax=Qmax, θ=θ, σ=σ)
    du[9] = Vrx
    du[10] = -(α + β)*Vrx + α*β*(-Vr + Ir)
    
    Ie = nu_es*sigmoid(hVs; Qmax=Qmax, θ=θ, σ=σ)
    du[1] = Vex
    du[2] = -(α + β)*Vex + α*β*(-Ve + nu_ee*Φe + nu_ei*sigmoid(Vi; Qmax=Qmax, θ=θ, σ=σ) + Ie)

    du[3] = Φex
    du[4] = -2*γ*Φex - γ^2*Φe + γ^2*sigmoid(Ve; Qmax=Qmax, θ=θ, σ=σ)

    Ii = nu_is*sigmoid(hVs; Qmax=Qmax, θ=θ, σ=σ)
    du[5] = Vix
    du[6] = -(α + β)*Vix + α*β*(-Vi + nu_ie*Φe + nu_ii*sigmoid(Vi; Qmax=Qmax, θ=θ, σ=σ) + Ii)
    
end

# Diffusion function
function g!(du, u, h, p, t)
    fill!(du, 0.0)
end

# Model parameters
p = (
    α = 83.33, β = 4 * 83.33, γ = 116.0, τ = 80e-3 / 2,
    Qmax = 340.0, θ = 12.92e-3, σ = 3.8e-3, Qmax_r = 340.0, σ_r = 3.8e-3,
    Φn0 = 1.0, Φn = 5e-4,
    nu_ee = 3.03e-3, nu_ei = -6.0e-3, nu_es = 2.06e-3,
    nu_ie = 3.03e-3, nu_ii = -6.0e-3, nu_is = 2.06e-3,
    nu_se = 2.18e-3, nu_sr = -0.83e-3, nu_sn = 0.98e-3,
    nu_re = 0.33e-3, nu_rs = 0.03e-3
)

# Time span
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4

# Define the noise function
n_points = Int((tspan[2]-tspan[1]) * (1/dts))
t_values = range(tspan[1], tspan[2], length=n_points)
sensory_input_expr_func_str = "t -> p.nu_sn * sqrt(p.Φn) * 100. * randn() + p.nu_sn*p.Φn0"
sensory_input_func = eval(Meta.parse(sensory_input_expr_func_str))
s_values = [Base.invokelatest(sensory_input_func, it) for it in t_values]
S = LinearInterpolation(t_values, s_values)

# join params p with S -> S 
p = merge(p, (S = S,))

# Initial condition
u0 = [0.0006344, 0.0, 3.175, 0.0, 0.0006344, 0.0, -0.003234, 0.0, 0.005676, 0.0]

# History function for delays
h(p, t) = u0

# Build SDDE problem
prob = SDDEProblem(f!, g!, u0, h, tspan, p, constant_lags=(p.τ,))
sol = solve(prob, EulerHeun(), dt=dt, saveat=dts)

# Extract results
signal_cols = [:Ve, :Vex, :Φe, :Φex, :Vi, :Vix, :Vs, :Vsx, :Vr, :Vrx]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, signal_cols[3])
model_plot = plot_brain_source_results(brain_df; model_name="Robinson-Rennie-Wright")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end