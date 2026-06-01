include(isfile(joinpath(@__DIR__, "utils_plot.jl")) ? joinpath(@__DIR__, "utils_plot.jl") : joinpath(@__DIR__, "models", "utils_plot.jl"))

# -----------------------------------------------------------------------------
# Reduced Wong-Wang model
# Brain source: synaptic gating variable S, matching the canonical ENEEGMA node
# definition.
# -----------------------------------------------------------------------------

function H(x, a, b, d)
    num = a * x - b
    denom = 1 - exp(-d * num)
    abs(denom) < 1e-12 && return 1 / d
    return num / denom
end

function f!(du, u, p, t)
    S = u[1]
    x = p.w * p.J_N * S + p.I_0
    du[1] = (-S / p.tau_s) + (1 - S) * p.gamma * H(x, p.a, p.b, p.d)
end

function g!(du, u, p, t)
    du[1] = p.sigma
end

p = (
    tau_s = 0.1,     # [s] NMDA decay (100 ms)
    gamma = 0.641,   # gain (TVB: 0.641 / 1000)
    w = 0.9,         # recurrent weight
    J_N = 0.2609,    # synaptic efficacy
    I_0 = 0.32,      # baseline input
    a = 270.0,       # gain of sigmoid [1/nC]
    b = 108.0,       # threshold [Hz]
    d = 0.154,       # steepness [ms]
    sigma = 0.02     # noise amplitude
)

u0 = [0.1]
tspan = (0.0, 60.0)
dts = 1e-3
dt = 1e-4
solver = EM()

prob = SDEProblem(f!, g!, u0, tspan, p)
sol = solve(prob, solver, dt=dt, saveat=dts)

signal_cols = [:S]
sol_df = DataFrame(
    merge(
        Dict(:time => sol.t),
        Dict(signal_cols[i] => [u[i] for u in sol.u] for i in 1:length(signal_cols))
    )
)
brain_df = brain_source_dataframe(sol_df, :S)
model_plot = plot_brain_source_results(brain_df; model_name="Reduced Wong-Wang")

if abspath(PROGRAM_FILE) == @__FILE__
    gui()
    println("Code successfully executed. Press Enter to exit the plot...")
    readline()
end