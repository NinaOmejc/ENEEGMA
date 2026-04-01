function get_input_dynamics(output_type::Symbol=:dict)
    d = Dict(
        "none" => skip_input_dynamics,
        "linear_kernel" => linear_kernel,
        "exp_kernel" => exp_kernel,
        "gating_kinetics" => gating_kinetics,
        "second_order_kernel" => second_order_kernel,
        "poly_kernel" => poly_kernel,
        "slowfast_piecewise_poly_kernel" => slowfast_piecewise_poly_kernel,
        "voltage_gated_dynamics" => voltage_gated_dynamics,
    )
    return output_type === :dict   ? d :
           output_type === :keys   ? collect(keys(d)) :
           output_type === :values ? collect(values(d)) :
           throw(ArgumentError("Unsupported output_type=$(output_type). Use :dict, :keys, or :values."))
end

function skip_input_dynamics(pop::Population, input_idx::Int)
    # Add a placeholder variable for inputs. This is crucial for substitution logic.
    vars = VarSet()
    var = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="input_placeholder", eq_idx=0, parent_pop=pop)
    add_var!(vars, var)
    return Vector{Equation}(), vars, ParamSet()
end


function linear_kernel(pop::Population, input_idx::Int=1)
    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)",
        eq_idx=1, parent_pop=pop,
        sends_interpop_output=true,
        gets_internode_input=pop.build_setts.gets_internode_input,
        gets_sensory_input=pop.build_setts.gets_sensory_input,
        gets_interpop_input=true,
        gets_additive_noise=true)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)",
        type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, inputs])

    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :offset, pop; description="baseline drive (additive bias)")
    input_dynamics_params = ParamSet([c1])

    input_dynamics = [
        D(symbol(x1)) ~ c1.symbol + symbol(inputs)
    ]
    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end


function exp_kernel(pop::Population, input_idx::Int=1)
    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
                  gets_sensory_input=pop.build_setts.gets_sensory_input,
                  gets_internode_input=pop.build_setts.gets_internode_input,
                  gets_interpop_input=true,
                  gets_additive_noise=true,
                  sends_internode_output=pop.build_setts.sends_internode_output,
                  sends_interpop_output=true)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x, inputs])

    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :time_constant,
               pop; tunable=true, description="tau (time constant)")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :damping,
               pop; tunable=true, description="")
    input_dynamics_params = ParamSet([c1, c2])

    input_dynamics = [ D(symbol(x)) ~ (1 / c1.symbol) * (symbol(inputs) - c2.symbol * symbol(x)) ]
    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end

function second_order_kernel(pop::Population, input_idx::Int=1)
    # is a generalization of several 2nd order kernels (alpha, biexp, oscillatory)

    hv = pop.build_setts.highest_var_idx
    hp = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(hv + 1)",
                   eq_idx=1, parent_pop=pop,
                   sends_internode_output=pop.build_setts.sends_internode_output,
                   sends_interpop_output=true)
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(hv + 2)",
                   eq_idx=2, parent_pop=pop,
                   gets_sensory_input=pop.build_setts.gets_sensory_input,
                   gets_internode_input=pop.build_setts.gets_internode_input,
                   gets_interpop_input=true,
                   gets_additive_noise=true,
                   sends_intrapop_output=true)

    I  = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)",
                  type="aux", eq_idx=0, parent_pop=pop)
    vars = VarSet([x1, x2, I])

    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(hp + 1)", :frequency, pop;
              description="natural frequency ω (s⁻¹)")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(hp + 2)", :population_coupling, pop;
              description="input coupling κ (can be signed)")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(hp + 3)", :damping, pop;
              description="damping ratio ζ")


    params = ParamSet([c1, c2, c3])

    eqs = [
        D(symbol(x1)) ~ symbol(x2),
        D(symbol(x2)) ~ c1.symbol^2*(c2.symbol*symbol(I) - symbol(x1)) - 2*c3.symbol*c1.symbol*symbol(x2)
    ]
    return (eqs, vars, params)
end


function poly_kernel(pop::Population, input_idx::Int=1)
    # poly_eq must be a comma-separated string like "x2, x1 * x2"
    poly_eq = pop.build_setts.input_dynamics_spec[input_idx]
    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    # State vars and input (mirror flags used in poly2_kernel)
    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
        sends_interpop_output=true,
        sends_internode_output=pop.build_setts.sends_internode_output)
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", eq_idx=2, parent_pop=pop,
        gets_internode_input=pop.build_setts.gets_internode_input, 
        gets_sensory_input=pop.build_setts.gets_sensory_input, 
        gets_interpop_input=true,
        gets_additive_noise=true)
    input = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, x2, input])

    # Parse "eq1, eq2" (trim and remove any parentheses added for formatting)
    eqs = strip.(split(poly_eq, ','))
    length(eqs) == 2 || throw(ArgumentError("poly_kernel expects two polynomials separated by ',' but got: '$poly_eq'"))
    eqs = replace.(eqs, '(' => ' ', ')' => ' ')

    params = Param[]
    next_idx = Ref(1)
    mkparam(desc::AbstractString) = begin
        p = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + next_idx[])", :poly_coeff,
                  pop; description=desc)
        push!(params, p)
        next_idx[] += 1
        p
    end

    # Build sum of products for an equation string like "x2 + x1 * x2"
    function build_sum(eqstr::String)
        isempty(strip(eqstr)) && return Num(0)
        terms = split(eqstr, '+')
        rhs = Num(0)
        for t in terms
            t = strip(t)
            isempty(t) && continue
            factors = split(t, '*')
            prod = Num(1)
            for f in factors
                f = strip(f)
                if f == "x1"
                    prod *= symbol(x1)
                elseif f == "x2"
                    prod *= symbol(x2)
                elseif isempty(f)
                    continue
                else
                    throw(ArgumentError("poly_kernel: unknown token '$f' in term '$t'"))
                end
            end
            c = mkparam("poly term")
            rhs += c.symbol * prod
        end
        return rhs
    end

    c1 = mkparam("const term")
    c2 = mkparam("const term")

    rhs1 = c1.symbol + build_sum(eqs[1])
    rhs2 = c2.symbol + build_sum(eqs[2])

    # For now, add input only to the second equation
    c1_in = mkparam("poly input scaling")
    c2_in = mkparam("poly input scaling")
    rhs1 = rhs1 + c1_in.symbol * symbol(input)
    rhs2 = rhs2 + c2_in.symbol * symbol(input)

    input_dynamics_params = ParamSet(params)
    input_dynamics = [
        D(symbol(x1)) ~ rhs1,
        D(symbol(x2)) ~ rhs2
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end

function gating_kinetics(pop::Population, input_idx::Int=1)
    # synaptic gating kinetics dynamics (e.g. for NMDA receptors)
    # dx/dt = -(1 / c1) * x + c2 * (1 - x) * inputs

    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
        gets_internode_input=pop.build_setts.gets_internode_input, gets_interpop_input=true,
        sends_interpop_output=true, gets_sensory_input=pop.build_setts.gets_sensory_input)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x, inputs])
    
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :time_constant,
               pop; description="tau (time constant)")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :gain,
               pop; description="nonlinearity scaling")
    input_dynamics_params = ParamSet([c1, c2])

    input_dynamics = [
        D(symbol(x)) ~ -(1 / c1.symbol) * symbol(x) + c2.symbol * (1 - symbol(x)) * symbol(inputs)
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end

function voltage_gated_dynamics(pop::Population, input_idx::Int=1)

    hv = pop.build_setts.highest_var_idx
    hp = pop.build_setts.highest_param_idx

    # State variables
    V = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(hv+1)",
                 eq_idx=1, parent_pop=pop,
                 gets_sensory_input=true,
                 sends_interpop_output=true)

    W = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(hv+2)",
                 eq_idx=2, parent_pop=pop)

    Z = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(hv+3)",
                 eq_idx=3, parent_pop=pop)

    input = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)",
                     type="aux", eq_idx=0, parent_pop=pop)

    vars = VarSet([V,W,Z, input])

    # Parameters (follow TVB naming exactly)
    p = Dict(
        :gCa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+1)",  :gain,  pop),
        :gK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+2)",  :gain,  pop),
        :gL  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+3)",  :gain,  pop),
        :gNa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+4)",  :gain,  pop),

        :VCa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+5)",  :offset, pop),
        :VK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+6)",  :offset, pop),
        :VL  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+7)",  :offset, pop),
        :VNa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+8)",  :offset, pop),

        :TCa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+9)", :offset, pop),
        :dCa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+10)",:rate,   pop),
        :TNa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+11)",:offset, pop),
        :dNa => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+12)",:rate,   pop),
        :TK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+13)",:offset, pop),
        :dK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+14)",:rate,   pop),

        :VT  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+15)",:offset, pop),
        :dV  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+16)",:rate,   pop),
        :ZT  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+17)",:offset, pop),
        :dZ  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+18)",:rate,   pop),
        
        :aee => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+19)",:population_coupling, pop),
        :aei => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+20)",:population_coupling, pop),
        :aie => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+21)",:population_coupling, pop),
        :ane => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+22)",:population_coupling, pop),
        :ani => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+23)",:population_coupling, pop),

        :C    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+24)",:probability, pop),
        :rNMDA=> Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+25)",:population_coupling, pop),

        :phi   => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+26)",:rate, pop),
        :tauK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+27)",:time_constant, pop),
        :b     => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+28)",:gain, pop),

        :QVmax => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+29)",:gain, pop),
        :QZmax => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+30)",:gain, pop),
    
    )

    params = ParamSet(collect(values(p)))

    # activation functions
    function tanh_sigmoid(x,T,d)
        return 0.5*(1 + tanh((x - T) / d))
    end

    mCa = tanh_sigmoid(symbol(V), p[:TCa].symbol, p[:dCa].symbol)
    mNa = tanh_sigmoid(symbol(V), p[:TNa].symbol, p[:dNa].symbol)
    mK  = tanh_sigmoid(symbol(V), p[:TK].symbol,  p[:dK].symbol)

    QV  = p[:QVmax].symbol * tanh_sigmoid(symbol(V), p[:VT].symbol, p[:dV].symbol)
    QZ  = p[:QZmax].symbol * tanh_sigmoid(symbol(Z), p[:ZT].symbol, p[:dZ].symbol)

    # currents
    I_Ca = (p[:gCa].symbol + (1 - p[:C].symbol)*p[:rNMDA].symbol*p[:aee].symbol*QV) * mCa * (symbol(V) - p[:VCa].symbol)
    I_K  = p[:gK].symbol * symbol(W) * (symbol(V) - p[:VK].symbol)
    I_L  = p[:gL].symbol * (symbol(V) - p[:VL].symbol)
    I_Na = (p[:gNa].symbol*mNa + (1-p[:C].symbol)*p[:aee].symbol*QV) * (symbol(V) - p[:VNa].symbol)
    I_IE = p[:aie].symbol * symbol(Z) * QZ
    I_ext = p[:ane].symbol * symbol(input)

    eqV = D(symbol(V)) ~ -(I_Ca + I_K + I_L + I_Na + I_IE) + I_ext
    eqW = D(symbol(W)) ~ p[:phi].symbol*(mK - symbol(W)) / p[:tauK].symbol
    eqZ = D(symbol(Z)) ~ p[:b].symbol*( p[:ani].symbol*symbol(input) + p[:aei].symbol*symbol(V)*QV )

    return ([eqV,eqW,eqZ], vars, params)
end



function slowfast_piecewise_poly_kernel(pop::Population, input_idx::Int=1)
    # Generic 3D slow–fast, piecewise-polynomial kernel.
    #
    # States:
    #   x :: fast variable   (e.g. x1 or x2)
    #   y :: fast recovery   (e.g. y1 or y2)
    #   w :: slow variable   (e.g. z or g)
    #
    # With appropriate parameter choices, this can reproduce both
    # epileptor_pop1_dynamics (x1,y1,z) and epileptor_pop2_dynamics (x2,y2,g).

    highest_var_idx   = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    # --- State variables and input -----------------------------------------
    x = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)",
                 eq_idx=1,
                 parent_pop=pop,
                 gets_sensory_input=pop.build_setts.gets_sensory_input,
                 gets_internode_input=pop.build_setts.gets_internode_input,
                 gets_interpop_input=true,
                 gets_additive_noise=true,
                 sends_internode_output=pop.build_setts.sends_internode_output,
                 sends_interpop_output=true)

    y = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)",
                 eq_idx=2,
                 parent_pop=pop,
                 gets_additive_noise=true)

    w = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 3)",
                 eq_idx=3,
                 parent_pop=pop,
                 sends_interpop_output=true)

    input = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)",
                     type="aux",
                     eq_idx=0,
                     parent_pop=pop)

    input_dynamics_vars = VarSet([x, y, w, input])

    # --- Parameters ---------------------------------------------------------
    # A compact vector of parameters, following your poly2gen_kernel style.
    c = [Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + i)",
               :gain, pop)
         for i in 1:31]
    input_dynamics_params = ParamSet(c)
    cs = [p.symbol for p in c]

    # Unpack for readability
    (θ_fast,  a0,  a1,  a2,  a3,  b0,  b1,  b2,
     θ_slow, sL0, sL1, sL2, sL3, sR0, sR1, sR2, sR3,
     A1, A2, A3, A4, A5, A6,
     B1, B2, B3, B4,
     C1, C2, C3, C4) = cs

    # --- Piecewise-polynomial building blocks ------------------------------
    # Fast piecewise term f_fast(x,w):
    #   if x < θ_fast:  cubic in x
    #   else          : (b0 + b1*w + b2*w^2) * x
    #
    # Epileptor-pop1:
    #   f1(x1,z) = a*x1^3 - b*x1^2           (x1 < 0)
    #             -(s + 0.6(z-4)^2)*x1      (x1 ≥ 0)
    # can be recovered by appropriate (θ_fast,a*,b*,w=z).
    f_fast(x::Num, w::Num) = ifelse(
        x < θ_fast,
        a0 + a1*x + a2*x^2 + a3*x^3,
        (b0 + b1*w + b2*w^2) * x
    )

    # Slow piecewise term f_slow(x):
    #   if x < θ_slow:  cubic in x
    #   else          : cubic in x
    #
    # Epileptor-pop2:
    #   f2(x2) = 0                   (x2 < -0.25)
    #           aa*(x2 + 0.25)       (x2 ≥ -0.25)
    # fits into this by choosing θ_slow = -0.25, sL* = 0, sR* linear.
    f_slow(x::Num) = ifelse(
        x < θ_slow,
        sL0 + sL1*x + sL2*x^2 + sL3*x^3,
        sR0 + sR1*x + sR2*x^2 + sR3*x^3
    )

    # --- Symbols ------------------------------------------------------------
    X = symbol(x)
    Y = symbol(y)
    W = symbol(w)
    U = symbol(input)

    # --- ODE system ---------------------------------------------------------
    #
    # Generic 3D slow–fast, piecewise-polynomial form:
    #
    #   dX/dt = A1*Y + A2*W + A3*U + A4 + A5*X + A6*X^3 - f_fast(X,W)
    #   dY/dt = B1*Y + B2     + B3*X + B4*X^2          - f_slow(X)
    #   dW/dt = C1*(C2*(X - C3) - W)
    #
    # Epileptor-pop1 (x1,y1,z):
    #   - choose parameters so that:
    #       dX ≈ y1 - z - f1(x1,z) + I1 + U
    #       dY ≈ c - d x1^2 - y1
    #       dW ≈ r ( s (x1 - x0) - z )
    #
    # Epileptor-pop2 (x2,y2,g):
    #   - choose parameters so that:
    #       dX ≈ -y2 + x2 - x2^3 + I2 + U
    #       dY ≈ (-y2 + f2(x2)) / τ
    #       dW ≈ -0.01 (g - 0.1 * X_drv)
    #     where X_drv can be realised either as the local X or as an
    #     external drive passed through U and C4.

    input_dynamics = [
        D(X) ~ A1*Y + A2*W + A3*U + A4 + A5*X + A6*X^3 - f_fast(X, W),
        D(Y) ~ B1*Y + B2     + B3*X + B4*X^2          - f_slow(X),
        D(W) ~ C1*(C2*(X - C3) - W) + C4*U
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end



##############################################
## Legacy kernels (to be deprecated)
###############################################

function alpha_kernel(pop::Population, input_idx::Int=1)
    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x11 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", 
                    eq_idx=1, 
                    parent_pop=pop,
                    sends_internode_output=pop.build_setts.sends_internode_output,
                    sends_interpop_output=true)
    x12 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", 
                    eq_idx=2, 
                    parent_pop=pop,
                    gets_sensory_input=pop.build_setts.gets_sensory_input,
                    gets_internode_input=pop.build_setts.gets_internode_input,
                    gets_interpop_input=true,
                    gets_additive_noise=true,
                    sends_intrapop_output=true)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x11, x12, inputs])

    c11 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :gain,
                pop; description="")
    c12 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :frequency,
                pop; description="")
    input_dynamics_params = ParamSet([c11, c12])

    input_dynamics = [
        D(symbol(x11)) ~ symbol(x12),
        D(symbol(x12)) ~ c11.symbol*c12.symbol*symbol(inputs) - c12.symbol^2 * symbol(x11) - 2 * c12.symbol * symbol(x12)
    ]
    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end

function biexp_kernel(pop::Population, input_idx::Int=1)
    # dx/dt = y
    # dy/dt = 1/(α*β) * (input - x) - (1/α + 1/β) * y 

    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x11 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
        sends_internode_output=pop.build_setts.sends_internode_output, 
        sends_interpop_output=true)
    x12 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", eq_idx=2, parent_pop=pop,
        gets_sensory_input=pop.build_setts.gets_sensory_input, 
        gets_internode_input=pop.build_setts.gets_internode_input,
        gets_interpop_input=true, 
        gets_additive_noise=true)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x11, x12, inputs])

    c11 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :time_constant,
                pop; description="Time constant, α")
    c12 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :time_constant,
                pop; description="Time constant, β")
    input_dynamics_params = ParamSet([c11, c12])

    input_dynamics = [
        D(symbol(x11)) ~ symbol(x12),
        D(symbol(x12)) ~ (c11.symbol * c12.symbol) * (symbol(inputs) - symbol(x11)) - (c11.symbol + c12.symbol) * symbol(x12)
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end


function oscillatory_kernel(pop::Population, input_idx::Int=1)
    # dx1/dt = x2
    # dx2/dt = -2ζω*x2 - ω²*x1 + ω²*I

    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop, sends_interpop_output=true)
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", eq_idx=2, parent_pop=pop,
        gets_internode_input=pop.build_setts.gets_internode_input, gets_interpop_input=true,
        gets_sensory_input=pop.build_setts.gets_sensory_input, gets_additive_noise=true)
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, x2, inputs])

    ζ = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :damping,
               pop; description="Damping ratio")
    ω = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :frequency,
               pop; description="Natural frequency")
    input_dynamics_params = ParamSet([ζ, ω])

    input_dynamics = [
        D(symbol(x1)) ~ symbol(x2),
        D(symbol(x2)) ~ ω.symbol^2*symbol(inputs) - 2*ζ.symbol*ω.symbol*symbol(x2) - ω.symbol^2*symbol(x1)
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end


function poly2_kernel(pop::Population, input_idx::Int=1)

    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
        sends_interpop_output=true, 
        gets_additive_noise=true)
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", eq_idx=2, parent_pop=pop,
        gets_interpop_input=true, 
        gets_internode_input=pop.build_setts.gets_internode_input,
        gets_sensory_input=pop.build_setts.gets_sensory_input)
    input = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", eq_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, x2, input])
    
    c = [Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + i)", :poly_coeff, pop) for i in 1:22]
    input_dynamics_params = ParamSet(c)
    cs = [p.symbol for p in c]

    input_dynamics = [
        D(symbol(x1)) ~ cs[1]*symbol(input) + cs[2] + cs[3]*symbol(x1) + cs[4]*symbol(x2) + cs[5]*symbol(x1)^2 + 
                        cs[6]*symbol(x1)*symbol(x2) + cs[7]*symbol(x2)^2 + cs[8]*symbol(x2)^3 + cs[9]*symbol(x1)^3 + 
                        cs[10]*symbol(x1)^2*symbol(x2) + cs[11]*symbol(x1)*symbol(x2)^2,
        D(symbol(x2)) ~ cs[12]*symbol(input) + cs[13] + cs[14]*symbol(x1) + cs[15]*symbol(x2) + cs[16]*symbol(x1)^2 + 
                        cs[17]*symbol(x1)*symbol(x2) + cs[18]*symbol(x2)^2 + cs[19]*symbol(x2)^3 + cs[20]*symbol(x1)^3 + 
                        cs[21]*symbol(x1)^2*symbol(x2) + cs[22]*symbol(x1)*symbol(x2)^2
        ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end


function lumped_voltage_gate_dynamics(pop::Population, input_idx::Int=1)
    hv = pop.build_setts.highest_var_idx
    hp = pop.build_setts.highest_param_idx

    # --- State variables (V, W, Z) -----------------------------------------
    V = StateVar(; name = "$(pop.parent_node.name)₊x$(pop.id)$(hv+1)",
                 eq_idx = 1, parent_pop = pop,
                 gets_sensory_input = true)

    W = StateVar(; name = "$(pop.parent_node.name)₊x$(pop.id)$(hv+2)",
                 eq_idx = 2, parent_pop = pop)

    Z = StateVar(; name = "$(pop.parent_node.name)₊x$(pop.id)$(hv+3)",
                 eq_idx = 3, parent_pop = pop)

    input = ExtraVar(; name = "$(pop.parent_node.name)₊inputs$(input_idx)",
                     type = "aux", eq_idx = 0, parent_pop = pop)

    vars = VarSet([V, W, Z, input])

    # --- LUMPED parameters (generic voltage-gated pop) ----------------------
    # Leak + K:
    p = Dict(
        :gL    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+1)",  :gain,   pop; description="Leak conductance"),
        :VL    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+2)",  :offset, pop; description="Leak reversal potential"),
        :gK    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+3)",  :gain,   pop; description="K+ conductance"),
        :VK    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+4)",  :offset, pop; description="K+ reversal potential"),

        :gExc  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+5)",  :gain, pop; description="effective excitatory gain"),
        :VExc  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+6)",  :offset, pop; description="effective excitatory reversal"),

        :gInh  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+7)",  :gain, pop; description="effective inhibitory gain"),

        :gExt  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+8)",  :gain, pop; description="external drive gain"),

        :phi   => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+9)",  :rate,  pop; description="adaptation rate"),
        :tauK  => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+10)", :time_constant, pop; description="adaptation time constant"),

        :b     => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+11)", :gain,    pop; description="Z drive gain"),

        :VT    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+12)", :offset, pop; description="V sigmoid midpoint"),
        :dV    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+13)", :rate,   pop; description="V sigmoid width"),
        :ZT    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+14)", :offset, pop; description="Z sigmoid midpoint"),
        :dZ    => Param("$(pop.parent_node.name)₊c$(pop.id)$(hp+15)", :rate,   pop; description="Z sigmoid width"),
    )

    params = ParamSet(collect(values(p)))

    # --- helper: generic tanh sigmoid --------------------------------------
    tanh_sigmoid(x, T, d) = 0.5 * (1 + tanh((x - T) / d))

    Vsym = symbol(V)
    Wsym = symbol(W)
    Zsym = symbol(Z)
    Isym = symbol(input)

    QV = tanh_sigmoid(Vsym, p[:VT].symbol, p[:dV].symbol)
    QZ = tanh_sigmoid(Zsym, p[:ZT].symbol, p[:dZ].symbol)

    # --- dynamics -----------------------------------------------------------
    # dV/dt = - I_leak - I_K - I_exc - I_inh + I_ext
    I_leak = p[:gL].symbol * (Vsym - p[:VL].symbol)
    I_K    = p[:gK].symbol * Wsym * (Vsym - p[:VK].symbol)
    I_exc  = p[:gExc].symbol * QV * (Vsym - p[:VExc].symbol)
    I_inh  = p[:gInh].symbol * Zsym * QZ
    I_ext  = p[:gExt].symbol * Isym

    eqV = D(Vsym) ~ -(I_leak + I_K + I_exc + I_inh) + I_ext

    # dW/dt = phi * (QV - W) / tauK  (K-like activation driven by V)
    eqW = D(Wsym) ~ p[:phi].symbol * (QV - Wsym) / p[:tauK].symbol

    # dZ/dt = b * (Isym + V * QV)  (simple interneuron drive)
    eqZ = D(Zsym) ~ p[:b].symbol * (Isym + Vsym * QV)

    return ([eqV, eqW, eqZ], vars, params)
end



function epileptor_pop1_dynamics(pop::Population, input_idx::Int=1)
    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)",
            system_idx=1,
            parent_pop=pop,
            gets_sensory_input=pop.build_setts.gets_sensory_input,
            gets_internode_input=pop.build_setts.gets_internode_input,
            gets_interpop_input=true,
            description="x1")
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", system_idx=2, parent_pop=pop, description="y1")
    x3 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 3)", system_idx=3, parent_pop=pop, sends_interpop_output=true, description="z")
    x4 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 4)", system_idx=4, parent_pop=pop, sends_interpop_output=true, description="g")
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)", type="aux", system_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, x2, x3, x4, inputs])

    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :gain, pop; description="a")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :gain, pop; description="b")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 3)", :gain, pop; description="s")
    c4 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 4)", :offset,        pop; description="c")
    c5 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 5)", :gain,        pop; description="d")
    c6 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 6)", :rate,         pop; description="r")
    c7 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 7)", :gain,         pop; description="s")
    c8 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 8)", :offset,         pop; description="x0")
    c9 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 9)", :gain,        pop; description="I1")
    input_dynamics_params = ParamSet([c1, c2, c3, c4, c5, c6, c7, c8, c9])

    f1_piecewise(x1::Num, x3::Num, a::Num, b::Num, s::Num) = ifelse(
        x1 < 0,
        a*x1^3 - b*x1^2,
        -x1*(s + 0.6*(x3 - 4)^2)
    )

    input_dynamics = [
        D(symbol(x1)) ~ symbol(x2) - symbol(x3) - f1_piecewise(symbol(x1), symbol(x3), c1.symbol, c2.symbol, c3.symbol) + symbol(inputs) + c9.symbol,
        D(symbol(x2)) ~ c4.symbol - c5.symbol*symbol(x1)^2 - symbol(x2),
        D(symbol(x3)) ~ c6.symbol*(c7.symbol*(symbol(x1) - c8.symbol) - symbol(x3)),
        D(symbol(x4)) ~ -0.01 * (symbol(x4) - 0.1*symbol(x1))
    ]

    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end

function epileptor_pop2_dynamics(pop::Population, input_idx::Int=1)

    highest_var_idx = pop.build_setts.highest_var_idx
    highest_param_idx = pop.build_setts.highest_param_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)",
        system_idx=1, parent_pop=pop,
        gets_sensory_input=pop.build_setts.gets_sensory_input,
        gets_internode_input=pop.build_setts.gets_internode_input,
        gets_interpop_input=true,
        sends_interpop_output=true,
        gets_additive_noise=true,
        description="x2")
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)",
        system_idx=2, parent_pop=pop,
        gets_additive_noise=true,
        description="y2")
    inputs = ExtraVar(; name="$(pop.parent_node.name)₊inputs$(input_idx)",
        type="aux", system_idx=0, parent_pop=pop)
    input_dynamics_vars = VarSet([x1, x2, inputs])

    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 1)", :gain, pop; description="aa")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 2)", :time_constant,         pop; description="tau")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_param_idx + 3)", :gain, pop; description="I2")
    input_dynamics_params = ParamSet([c1, c2, c3])

    f2_piecewise(x1::Num, a::Num) = ifelse(
        x1 < -0.25,
        0.0,
        a*(x1 + 0.25)
    )

    input_dynamics = [
        D(symbol(x1)) ~ -symbol(x2) + symbol(x1) - symbol(x1)^3 + c3.symbol + symbol(inputs),
        D(symbol(x2)) ~ (-symbol(x2) + f2_piecewise(symbol(x1), c1.symbol))/c2.symbol
    ]
    
    return (input_dynamics, input_dynamics_vars, input_dynamics_params)
end