struct NoGradOptSol
    u::Vector{Float64}
    minimum::Float64
    retcode::Symbol
end

mutable struct NoGradState
    iter::Int
    u::Vector{Float64}
end


function run_adam_nograd(
    optfun::OptimizationFunction,   # your existing optfun (already wraps reparam)
    args::NamedTuple,
    u0::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    os::OptimizationSettings,
    callback_fun::Function
)::NoGradOptSol

    oz = os.optimizer_settings

    lr       = hasproperty(oz, :lr)       ? Float64(oz.lr)       : 1e-2
    iters    = hasproperty(oz, :maxiters) ? Int(oz.maxiters)     : os.maxiters
    fd_eps   = hasproperty(oz, :fd_eps)   ? Float64(oz.fd_eps)   : 1e-3
    avg_dirs = hasproperty(oz, :avg_dirs) ? Int(oz.avg_dirs)     : 2
    seed     = hasproperty(oz, :seed)     ? Int(oz.seed)         : 1

    rng = Random.MersenneTwister(seed)

    # loss function consistent with Optimization.solve
    loss_u = (u::Vector{Float64}) -> optfun(u, args)

    u = copy(u0)
    @inbounds for i in eachindex(u)
        u[i] = clamp(u[i], lb[i], ub[i])
    end

    # Adam moments
    m = zeros(length(u))
    v = zeros(length(u))

    best_u = copy(u)
    best_f = loss_u(u)

    state = NoGradState(0, copy(u))

    # initial callback (optional, but matches your logging behavior)
    callback_fun(state, best_f)

    for t in 1:iters
        state.iter = t
        state.u .= u

        # compute current loss
        f0 = loss_u(u)

        # estimate direction (your simple finite-diff style)
        ghat = zeros(length(u))
        δ = similar(u)

        for _ in 1:avg_dirs
            @inbounds for i in eachindex(u)
                δ[i] = randn(rng)
            end
            u1 = u .+ fd_eps .* δ
            @inbounds for i in eachindex(u1)
                u1[i] = clamp(u1[i], lb[i], ub[i])
            end
            f1 = loss_u(u1)
            @inbounds for i in eachindex(u)
                ghat[i] += (f1 - f0) * δ[i]
            end
        end
        @inbounds for i in eachindex(ghat)
            ghat[i] /= avg_dirs
        end

        # Adam update
        β1 = 0.9
        β2 = 0.999
        ϵ  = 1e-8

        m .= β1 .* m .+ (1 - β1) .* ghat
        v .= β2 .* v .+ (1 - β2) .* (ghat .^ 2)

        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)

        u .-= lr .* m̂ ./ (sqrt.(v̂) .+ ϵ)

        # bounds
        @inbounds for i in eachindex(u)
            u[i] = clamp(u[i], lb[i], ub[i])
        end

        # track best
        if f0 < best_f
            best_f = f0
            best_u .= u
        end

        # callback (your logger/printing/time limit lives here)
        stop = callback_fun(state, f0)
        if stop
            return NoGradOptSol(best_u, best_f, :Terminated)
        end
    end

    return NoGradOptSol(best_u, best_f, :Success)
end


function adam_nograd(
    loss, u0;
    lr=1e-2, β1=0.9, β2=0.999, ϵ=1e-8,
    iters=1000
)
    u = copy(u0)
    m = zeros(length(u))
    v = zeros(length(u))

    best_u = copy(u)
    best_loss = Inf

    for t in 1:iters
        # random perturbation direction
        δ = randn(length(u))
        L0 = loss(u)
        L1 = loss(u .+ 1e-4 .* δ)

        ĝ = (L1 - L0) .* δ  # directional estimate

        m .= β1 .* m .+ (1 - β1) .* ĝ
        v .= β2 .* v .+ (1 - β2) .* (ĝ .^ 2)

        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)

        u .-= lr .* m̂ ./ (sqrt.(v̂) .+ ϵ)

        if L0 < best_loss
            best_loss = L0
            best_u .= u
        end
    end

    return best_u, best_loss
end

#= 
"optimization_settings": {
  "method": "ADAM_NOGRAD",
  "maxiters": 400,
  "optimizer_settings": {
    "lr": 0.01,
    "fd_eps": 0.003,
    "avg_dirs": 2,
    "seed": 1
  }
} 
  
If it “jitters” without improving:
increase avg_dirs
increase fd_eps slightly
lower lr

=#