using StochasticDiffEq

include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Liley-Wright mean-field model
# Brain source: excitatory membrane potential Ve, corresponding to canonical x15.
# -----------------------------------------------------------------------------

# Parameters (same as params_map in run_LW)
# ───────────────────────── parameters in SECONDS ────────────────────────────

P = (S_e_max=0.5*1e3, # 0.5 ms⁻¹ → 500 s⁻¹
     S_i_max=0.5*1e3,
     h_e_r=-70.0, h_i_r=-70.0,
     μ_e=-50.0, μ_i=-50.0,
     σ_e=5.0, σ_i=5.0,
     τ_e=94/1e3, τ_i=42/1e3,  # 94 ms → 0.094 s, 42 ms → 0.042 s
     h_ee_eq=45.0, h_ei_eq = 45.0,
     h_ie_eq=-90.0, h_ii_eq=-90.0,
     Γ_ee=0.71*ℯ, Γ_ei=0.71*ℯ, Γ_ie=0.71*ℯ, Γ_ii=0.71*ℯ,
     γ_ee=0.3*1e3, γ_ei=0.3*1e3, γ_ie=0.065*1e3, γ_ii=0.065*1e3, # 0.3 ms⁻¹ → 300 s⁻¹
     p_ee=3.460*1e3, p_ei=3.460*1e3, # orig: p_ei=5.070*1e3, #p_ee=3.460*1e3
     p_ie=0.0, p_ii=0.0,  # 3.46 ms⁻¹ → 3460 s⁻¹
     p_ee_sd=1000.0, p_ei_sd=0.0,
     N_ee_b=3000., N_ei_b=3000.,
     N_ie_b=500., N_ii_b=500.
)


# ───────────────────── firing‐rate (sigmoid) functions ──────────────────────
sigmoid(v, S_max, μ, σ) = S_max / (1 + exp(-sqrt(2)*(v - μ)/σ))

# ─────────────────────────── drift function f! ──────────────────────────────
function drift!(du, u, P, t)
    Ve, Vi, Iee, Jee, Iei, Jei, Iie, Jie, Iii, Jii = u

    ψ_ee = (P.h_ee_eq - Ve)/abs(P.h_ee_eq - P.h_e_r)
    ψ_ei = (P.h_ei_eq - Vi)/abs(P.h_ei_eq - P.h_i_r)
    ψ_ie = (P.h_ie_eq - Ve)/abs(P.h_ie_eq - P.h_e_r)
    ψ_ii = (P.h_ii_eq - Vi)/abs(P.h_ii_eq - P.h_i_r)

    A_ee = P.N_ee_b*sigmoid(Ve, P.S_e_max, P.μ_e, P.σ_e) + P.p_ee
    A_ei = P.N_ei_b*sigmoid(Ve, P.S_e_max, P.μ_e, P.σ_e) + P.p_ei
    A_ie = P.N_ie_b*sigmoid(Vi, P.S_i_max, P.μ_i, P.σ_i)
    A_ii = P.N_ii_b*sigmoid(Vi, P.S_i_max, P.μ_i, P.σ_i)

    du[1] = (P.h_e_r - Ve + ψ_ee*Iee + ψ_ie*Iie)/P.τ_e
    du[2] = (P.h_i_r - Vi + ψ_ei*Iei + ψ_ii*Iii)/P.τ_i

    du[3] = Jee;
    du[4] = -2P.γ_ee*Jee - P.γ_ee^2*Iee + P.γ_ee*P.Γ_ee*A_ee

    du[5] = Jei;
    du[6] = -2P.γ_ei*Jei - P.γ_ei^2*Iei + P.γ_ei*P.Γ_ei*A_ei
    
    du[7] = Jie;
    du[8] = -2P.γ_ie*Jie - P.γ_ie^2*Iie + P.γ_ie*P.Γ_ie*A_ie

    du[9] = Jii
    du[10]= -2P.γ_ii*Jii - P.γ_ii^2*Iii + P.γ_ii*P.Γ_ii*A_ii


    return nothing
end

# ───────────────────── diffusion (noise) function g! ────────────────────────
function diffusion!(dW, X, P, t)
    dW .= 0.0
    dW[4] = P.γ_ee * P.Γ_ee * P.p_ee_sd
end

# ───────────────────────── initial conditions ───────────────────────────────
Ve0, Vi0 = P.h_e_r, P.h_i_r
Iee0 = P.Γ_ee/P.γ_ee *(P.N_ee_b*sigmoid(Ve0, P.S_e_max, P.μ_e, P.σ_e) + P.p_ee)
Iei0 = P.Γ_ei/P.γ_ei *(P.N_ei_b*sigmoid(Ve0, P.S_e_max, P.μ_e, P.σ_e) + P.p_ei)
Iie0 = P.Γ_ie/P.γ_ie *(P.N_ie_b*sigmoid(Vi0, P.S_i_max, P.μ_i, P.σ_i))
Iii0 = P.Γ_ii/P.γ_ii *(P.N_ii_b*sigmoid(Vi0, P.S_i_max, P.μ_i, P.σ_i))

u0 = [Ve0, Vi0, Iee0, 0.0, Iei0, 0.0, Iie0, 0.0, Iii0, 0.0]

# ───────────────────────── simulation controls ──────────────────────────────
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4

prob = SDEProblem(drift!, diffusion!, u0, tspan, P)
sol  = solve(prob, EM(), dt = dt, saveat = dts)

# ──────────────────────── analysis & visualisation ─────────────────────────
t_sec = sol.t ./ 1_000              # convert ms → s
Ve    = -sol[1,:]                   # negative V_e as “EEG”

signal_cols = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :x1)
model_plot = plot_brain_source_results(brain_df; model_name="Liley-Wright")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end