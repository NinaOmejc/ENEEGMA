
###############################################################################
# Reduced Wong–Wang E–I (TVB-faithful) in Julia
#
# https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_library/tvb/simulator/models/wong_wang.py
# Brain source: excitatory synaptic gating variable S_e.
#
# Goals:
# - Match TVB model equations & parameter meanings as closely as possible
# - Use NamedTuple params + field access (no positional destructuring)
# - Support both:
#     (A) single node (coupling = 0)
#     (B) network-style coupling input via a user-supplied coupling function
#
# Notes vs TVB:
# - TVB uses ms internally (tau in ms, gamma in 1/ms, dt in ms).
# - Here we run in seconds. We convert TVB defaults consistently:
#     tau_e_ms -> tau_e_s = tau_e_ms/1000
#     gamma_e_per_ms -> gamma_e_per_s = gamma_e_per_ms*1000
###############################################################################

include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -------------------------
# Transfer function (TVB)
# -------------------------
# TVB:
#   x̃ = a*x - b
#   H = x̃ / (1 - exp(-d*x̃))
#
# Numerical guard for small denominator.
function h(x::Float64, a::Float64, b::Float64, d::Float64)
    x̃ = a * x - b
    denom = 1 - exp(-d * x̃)
    if abs(denom) < 1e-12
        # first-order approximation around 0:
        # 1 - exp(-d*x̃) ≈ d*x̃  -> H ≈ 1/d
        return 1 / d
    end
    return x̃ / denom
end

# -------------------------
# Coupling helpers
# -------------------------
# In TVB numba:
#   cc = G * J_N * c
#   x_e += cc
#   x_i += lamda * cc
#
# Here, `c` is a scalar "coupling input" for this node at time t.
# For a single node: c(t) = 0.0.
#
# For a network you can define your own c(t,u,p,t)->Float64.
default_coupling(u, p, t) = 0.0

# -------------------------
# Drift (TVB-faithful)
# -------------------------
function f!(du, u, p, t; coupling_func=default_coupling)
    S_e, S_i = u

    # network/local coupling input (scalar), TVB calls it "c"
    c = coupling_func(u, p, t)

    # TVB: cc = G * J_N * c
    cc = p.G * p.J_N * c

    # TVB: jnSe = J_N * S_e
    jnSe = p.J_N * S_e

    # TVB:
    # x_e = w_p * jnSe - J_i*S_i + W_e*I_o + cc
    x_e = p.w_p * jnSe - p.J_i * S_i + p.W_e * p.I_o + cc
    h_e = h(x_e, p.a_e, p.b_e, p.d_e)

    # dS_e = -(S_e/tau_e) + (1-S_e)*h_e*gamma_e
    du[1] = -(S_e / p.tau_e) + (1 - S_e) * h_e * p.gamma_e

    # TVB:
    # x_i = jnSe - S_i + W_i*I_o + lamda*cc
    x_i = jnSe - S_i + p.W_i * p.I_o + p.lamda * cc
    h_i = h(x_i, p.a_i, p.b_i, p.d_i)

    # dS_i = -(S_i/tau_i) + h_i*gamma_i
    du[2] = -(S_i / p.tau_i) + h_i * p.gamma_i

    return nothing
end

# -------------------------
# Diffusion (additive noise)
# -------------------------
# TVB adds noise at the simulator/integrator level, but your workflow uses SDEProblem.
# We keep it simple: dS = ... dt + σ dW
function g!(du, u, p, t)
    du[1] = p.sigma_e
    du[2] = p.sigma_i
    return nothing
end

# -------------------------
# TVB default parameters (converted to seconds)
# -------------------------
# TVB defaults (from the file you pasted):
#   a_e=310, b_e=125, d_e=0.160
#   gamma_e = 0.641/1000  (1/ms)
#   tau_e = 100 ms
#   w_p = 1.4
#   J_N = 0.15
#   W_e = 1.0
#   a_i=615, b_i=177, d_i=0.087
#   gamma_i = 1.0/1000    (1/ms)
#   tau_i = 10 ms   (note: TVB domain range in snippet is wrong, but default is 10)
#   J_i = 1.0
#   W_i = 0.7
#   I_o = 0.382
#   G = 2.0
#   lamda = 0.0
#
# Convert ms -> s and 1/ms -> 1/s:
#   tau_s = tau_ms / 1000
#   gamma_per_s = gamma_per_ms * 1000
function tvb_default_params(; sigma_e=0.0, sigma_i=0.0)
    return (
        # transfer (E)
        a_e = 310.0,
        b_e = 125.0,
        d_e = 0.160,

        # kinetics (E): TVB gamma_e=0.641/1000 1/ms -> 0.641 1/s
        gamma_e = 0.641,
        # tau_e=100 ms -> 0.1 s
        tau_e   = 0.100,

        # weights / currents
        w_p = 1.4,
        W_e = 1.0,
        J_N = 0.15,

        # transfer (I)
        a_i = 615.0,
        b_i = 177.0,
        d_i = 0.087,

        # kinetics (I): TVB gamma_i=1.0/1000 1/ms -> 1.0 1/s
        gamma_i = 1.0,
        # tau_i=10 ms -> 0.01 s
        tau_i   = 0.010,

        # weights / currents
        W_i = 0.7,
        J_i = 1.0,

        # external drive
        I_o = 0.382,

        # coupling
        G     = 2.0,
        lamda = 0.0,

        # noise (SDE)
        sigma_e = sigma_e,
        sigma_i = sigma_i,
    )
end

# -------------------------
# Simulation wrapper
# -------------------------
function simulate_tvb_ww(; p=tvb_default_params(),
                          u0=[0.1, 0.1],
                          tspan=(0.0, 40.0),
                          dt=1e-3,
                          saveat=1e-3,
                          coupling_func=default_coupling,
                          seed=0)

    Random.seed!(seed)

    # Wrap f! to pass coupling_func without global state
    fwrap!(du, u, p, t) = f!(du, u, p, t; coupling_func=coupling_func)

    prob = SDEProblem(fwrap!, g!, u0, tspan, p)
    sol = solve(prob, EM(); dt=dt, saveat=saveat)
    return sol
end

# -------------------------
# Example: single node, TVB defaults
# -------------------------
p_base = tvb_default_params(sigma_e=0.01, sigma_i=0.01)

sol = simulate_tvb_ww(p=p_base, tspan=(0.0, 60.0), dt=1e-4, saveat=1e-3, seed=1)

sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(:S_e => [u[1] for u in sol.u], :S_i => [u[2] for u in sol.u])
    )
)
sol_df = sol_df[sol_df.time .>= 10.0, :] # discard transient
brain_df = brain_source_dataframe(sol_df, :S_e)
model_plot = plot_brain_source_results(brain_df; model_name="Wong-Wang")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end