include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# ARM (Lopes da Silva et al., 1974) — linear thalamic E↔I loop with α-synapses
# Pop E: thalamo-cortical relay (excitatory)
# Pop I: thalamic reticular (inhibitory)
#
# Each population's mean membrane potential v_X obeys an α-filter ODE:
#    d v_X / dt = w_X
#    d w_X / dt = α_X^2 ( κ_X * u_X  - v_X ) - 2α_X * w_X
# where u_X is the summed synaptic drive (linearized), and κ_X is PSP gain.
#
# Coupling (linear, small-signal):
#   u_E =  P_E(t) - C_EI * v_I            (inhibition onto E)
#   u_I =  P_I(t) + C_IE * v_E            (excitation onto I)
#
# Stochastic drive: additive white noise on the acceleration terms (w'_X)
# -----------------------------------------------------------------------------

# Stochastic inputs (choose either constant or time-varying mean)
const P_E0 = 0.0
const P_I0 = 0.0
P_E(t) = P_E0
P_I(t) = P_I0

# Parameters
p = (
    αE = 70.0,      # 1/s  (alpha-synapse rate for E; time-to-peak ≈ 1/α)
    αI = 50.0,      # 1/s  (I is often slower)
    κE = 1.0,       # PSP gain E
    κI = 1.2,       # PSP gain I
    C_EI = 2.0,     # I -> E (inhibitory) coupling (linearized)
    C_IE = 1.6,     # E -> I (excitatory) coupling (linearized)
    σE = 1.0,       # noise intensity for E "acceleration"
    σI = 1.0        # noise intensity for I "acceleration"
)

# State: u = [vE, wE, vI, wI]
function f!(du, u, p, t)
    vE, wE, vI, wI = u
    uE = P_E(t) - p.C_EI * vI
    uI = P_I(t) + p.C_IE * vE

    du[1] = wE
    du[2] = p.αE^2 * (p.κE * uE - vE) - 2p.αE * wE
    du[3] = wI
    du[4] = p.αI^2 * (p.κI * uI - vI) - 2p.αI * wI
end

# Additive noise on the second derivatives (linear SDE)
function g!(du, u, p, t)
    du[1] = 0.0
    du[2] = p.σE
    du[3] = 0.0
    du[4] = p.σI
end

# Initial state
u0 = [0.0, 0.0, 0.0, 0.0]

# Simulation settings
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = SOSRI()           # SDE solver; Tsit5() also OK if σE=σI=0

prob = SDEProblem(f!, g!, u0, tspan, p)
sol  = solve(prob, solver, dt=dt, saveat=dts);

# Assemble dataframe
signal_cols = [:vE, :wE, :vI, :wI]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :vE)
model_plot = plot_brain_source_results(brain_df; model_name="Alpha Rhythm Model")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end