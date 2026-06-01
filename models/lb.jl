###############################################################################
# Larter–Breakspear (single column)
# https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_library/tvb/simulator/models/larter_breakspear.py
# Brain source: membrane potential V, matching the canonical ENEEGMA node definition.
###############################################################################
using StochasticDiffEq

include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))


# ----------------------------------------------------
# 1. Helper functions
# ----------------------------------------------------
sigmoid(x, T, δ)  = 0.5 * (1 + tanh((x - T)/δ))       # generic tanh sigmoids
m_inf(x, T, δ)    = sigmoid(x, T, δ)                   # channel gating (m_K, m_Na, m_Ca)
Q_rate(x, T, δ)   = sigmoid(x, T, δ)                   # firing-rate sigmoids (QV,QZ)

# ----------------------------------------------------
# 2. Core dynamical equations  (V, W, Z)
#    du = f(u,p,t)  with **named parameters** in the `p` NamedTuple
# ----------------------------------------------------
function f!(du, u, p, t)
    # unpack state
    V, W, Z = u

    # ---- fast channel activation ----
    m_Ca = m_inf(V, p.TCa, p.d_Ca)
    m_Na = m_inf(V, p.TNa, p.d_Na)
    m_K  = m_inf(V, p.TK,  p.d_K)

    # ---- population firing rates ----
    QV   = p.QV_max * Q_rate(V, p.VT, p.d_V)
    QZ   = p.QZ_max * Q_rate(Z, p.ZT, p.d_Z)

    # ---- membrane-potential derivative ----
    dV = -(p.gCa + (1 - p.C) * p.rNMDA * p.aee * QV) * m_Ca * (V - p.VCa)
    dV -=  p.gK  * W * (V - p.VK) + p.gL * (V - p.VL)
    dV -= (p.gNa * m_Na + (1 - p.C) * p.aee * QV)  * (V - p.VNa)
    dV -=  p.aie * Z * QZ
    dV +=  p.ane * p.I_ext(t)                             # external drive

    # ---- potassium activation variable ----
    dW = p.phi * (m_K - W) / p.tau_K

    # ---- inhibitory interneuron drive ----
    dZ = p.b * (p.ani * p.I_ext(t) + p.aei * V * QV)

    # scale everything if desired
    du[1] = p.t_scale * dV
    du[2] = p.t_scale * dW
    du[3] = p.t_scale * dZ
end

# Optional additive noise (set to zeros for pure ODE integration)
function g!(du, u, p, t)
    fill!(du, 0.0)
end

# ----------------------------------------------------
# 3. Parameter set  –  identical to TVB defaults
# ----------------------------------------------------
p = (;                       # NamedTuple – access with p.gCa etc.
    # conductances & reversal potentials
    gCa = 1.1,  gK = 2.0,  gL = 0.5,  gNa = 6.7,
    VCa = 1.0,  VK = -0.7, VL = -0.5, VNa = 0.53,

    # channel thresholds and widths
    TCa = -0.01, d_Ca = 0.15,
    TK  = 0.0,   d_K  = 0.3,
    TNa = 0.3,   d_Na = 0.15,

    # population-gain sigmoids
    VT  = 0.0,  d_V = 0.65,
    ZT  = 0.0,  d_Z = 0.7,
    QV_max = 1.0, QZ_max = 1.0,

    # synaptic strengths
    aee = 0.4, aei = 2.0, aie = 2.0, ane = 1.0, ani = 0.4,
    rNMDA = 0.25, C = 0.1,

    # time-constants / scaling
    tau_K = 1.0,  phi = 0.7,  b = 0.1,
    t_scale = 100., # **ms → s**

    # external input – here: constant, but you can swap in an arbitrary function
    I_ext = t -> 0.0000
)


# ----------------------------------------------------
# 4. Simulation setup
# ----------------------------------------------------
u0 = [-0.7, -0.2, 0.5]        # initial state (V,W,Z)
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4

prob  = SDEProblem(f!, g!, u0, tspan, p)   # replace with ODEProblem if g! ≡ 0
sol   = solve(prob, EM(), dt = dt, saveat = dts)

# ----------------------------------------------------
# 5. Quick post-processing & plotting (optional)
# ----------------------------------------------------

signal_cols = [:V, :W, :Z]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :V)
model_plot = plot_brain_source_results(brain_df; model_name="Larter-Breakspear")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end