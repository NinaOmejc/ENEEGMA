function list_canonical_node_models()::Vector{String}
    return ["WilsonCowan ('WC')", 
            "AlphaRhythm ('ARM')",
            "JansenRit ('JR')", 
            "Wendling ('W')",
            "MoranDavidFriston ('MDF')",
            "LileyWright ('LW')",
            "RobinsonRennieWrightCortex ('RRWC')", 
            "RobinsonRennieWrightThalamus ('RRWT')",
            "WongWang ('WW')",
            "WongWangReduced ('WWR')",
            "LarterBreakspear ('LB')",
            "HarmonicOscillator ('HO')",
            "Montbrio ('MPR')",
            "FitzHughNagumo ('FHN')",
            "VanDerPol ('VDP')",
            "StuartLandau ('SL')",
            "DuffingOscillator ('DO')",
            "PhaseDynamics ('PD')",
    ]
end


# Return vector of model identifiers. 
# style = :short  → ["WC","JR",...]
# style = :long   → ["WilsonCowan","JansenRit",...]
function list_canonical_node_models_codes(; style::Symbol = :short)
    raw = list_canonical_node_models()
    out = String[]
    for entry in raw
        m = match(r"^(.+?)\s*\('([^']+)'\)\s*$", entry)
        if m === nothing
            push!(out, entry)
        else
            long, short = strip.(m.captures)
            push!(out, style === :long ? long : short)
        end
    end
    return out
end

function get_canonical_node_model_info!(node::Node)
    model = node.build_setts.model
    
    # If model is a RuleTree (grammar-sampled), cannot use canonical model lookup
    # Return empty populations; will be built via configure_node_model! instead
    if model isa RuleTree
        return Population[], node
    end
    
    if model == "AlphaRhythm" || model == "ARM"
        return AlphaRhythm(node)
    elseif model == "WilsonCowan" || model == "WC"
        return WilsonCowan(node)
    elseif model == "RobinsonRennieWrightCortex" || model == "RRWC"
        return RobinsonRennieWrightCortex(node)
    elseif model == "RobinsonRennieWrightThalamus" || model == "RRWT"
        return RobinsonRennieWrightThalamus(node)
    elseif model == "JansenRit" || model == "JR"
        return JansenRit(node)
    elseif model == "Wendling" || model == "W"
        return Wendling(node)
    elseif model == "MoranDavidFriston" || model == "MDF"
        return MoranDavidFriston(node)
    elseif model == "LileyWright" || model == "LW"
        return LileyWright(node)
    elseif model == "WongWang" || model == "WW"
        return WongWang(node)
    elseif model == "WongWangReduced" || model == "WWR"
        return WongWangReduced(node)
    elseif model == "LarterBreakspear" || model == "LB"
        return LarterBreakspear(node)
    elseif model == "HarmonicOscillator" || model == "HO"
        return HarmonicOscillator(node)
    elseif model == "Montbrio" || model == "MPR"
        return Montbrio(node)
    elseif model == "FitzHughNagumo" || model == "FHN"
        return FitzHughNagumo(node)
    elseif model == "VanDerPol" || model == "VDP"
        return VanDerPol(node)     
    elseif model == "StuartLandau" || model == "SL"
        return StuartLandau(node)   
    elseif model == "DuffingOscillator" || model == "DO"
        return DuffingOscillator(node)
    elseif model == "PhaseDynamics" || model == "PD"
        return PhaseDynamics(node)
    else
        error("Unknown canonical node model: $(model). 
              Supported models are: $(list_canonical_node_models()).")
    end
end

function WilsonCowan(node::Node)
    pops = [Population(1, "E", node; input_dynamics_spec="exp_kernel",
                sensory_conn_func="baseline_sigmoid",
                internode_conn_func="baseline_sigmoid",
                sends_internode_output=true)
            Population(2, "I", node; input_dynamics_spec="exp_kernel",
                sensory_conn_func="baseline_sigmoid",
                internode_conn_func="none")]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("baseline_sigmoid", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 1.0),
        "$(node.name)₊x21" => (0.0, 1.0)
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # Population 1 (E) parameters
        "$(node.name)₊c11" => (4e-2, ),    # τ1 (tau_E)
        "$(node.name)₊c12" => (1.0, ),    # not used
        "$(node.name)₊c13" => (1.0, ),      # a
        "$(node.name)₊c14" => (5.0, ),     # θ
        "$(node.name)₊c15" => (1.0, ),      # not used
        "$(node.name)₊c16" => (16.0, ),   # cli_11 (E -> E strength)
        "$(node.name)₊c17" => (-26.0, ), # cli_12 (I -> E strength, inhibitory)

        # Population 2 (I) parameters
        "$(node.name)₊c21" => (4e-2, ),   # τ2 (tau_I)
        "$(node.name)₊c22" => (1.0, ),   # not used
        "$(node.name)₊c23" => (1.0, ),     # a
        "$(node.name)₊c24" => (20.0, ),  # θ
        "$(node.name)₊c25" => (1.0, ),     # not used
        "$(node.name)₊c26" => (20.0, ),  # cli_21 (E -> I strength)
        "$(node.name)₊c27" => (-1.0, ),  # cli_22 (I -> I strength, inhibitory)
    )

#=     node.build_setts.new_param_tunability = Dict{String, Bool}(
        "$(node.name)₊c11" => true,  # τ1 (tau_E)
        "$(node.name)₊c12" => false, # not used
        "$(node.name)₊c13" => false, # a
        "$(node.name)₊c14" => false, # θ
        "$(node.name)₊c15" => false,  # not used
        "$(node.name)₊c16" => true,  # cli_11 (E -> E strength)
        "$(node.name)₊c17" => true,  # cli_12 (I -> E strength, inhibitory)
        "$(node.name)₊c21" => true,  # τ2 (tau_I)
        "$(node.name)₊c22" => false, # not used
        "$(node.name)₊c23" => false, # a
        "$(node.name)₊c24" => false, # θ
        "$(node.name)₊c25" => false,  # not used
        "$(node.name)₊c26" => true,  # cli_21 (E -> I strength)
        "$(node.name)₊c27" => true,  # cli_22 (I -> I strength, inhibitory)
    ) =#

    return pops, node
end

function AlphaRhythm(node::Node)
    pops = [Population(1, "E", node; 
                input_dynamics_spec="second_order_kernel",
                sends_internode_output=true,
                sensory_conn_func="linear",
                internode_conn_func="linear",
                noise_dynamics_spec="stochastic",
                noise_dynamics="c"),
            Population(2, "I", node; 
                input_dynamics_spec="second_order_kernel",
                sensory_conn_func="linear",
                noise_dynamics_spec="stochastic",
                noise_dynamics="c")]
    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[2, 3]] .= "linear"
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 0.0),  # vE
        "$(node.name)₊x12" => (0.0, 0.0),  # wE
        "$(node.name)₊x21" => (0.0, 0.0),  # vI
        "$(node.name)₊x22" => (0.0, 0.0)   # wI
    )
    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # E population
        "$(node.name)₊c11" => (70.0, ), # αE; E alpha rate (time constant / resonance)
        "$(node.name)₊c12" => (-2.0, ),   # ≈ -κE*C_EI  (effective I→E gain)
        "$(node.name)₊c13" => (1.0, ),     # ≈ 1  (so that c13 = αE)
        "$(node.name)₊c14" => (1.0, ),     # σE (stochastic input amplitude)
        # I population
        "$(node.name)₊c21" => (50.0, ),  # αI
        "$(node.name)₊c22" => (1.92, ),    # ≈ κI*C_IE  (effective E→I gain)
        "$(node.name)₊c23" => (1.0, ),     # ≈ 1  (so that c23 = αI)
        "$(node.name)₊c24" => (1.0, )      # σI (stochastic input amplitude)
    )


#=     node.build_setts.new_param_tunability = Dict{String, Bool}(
        "$(node.name)₊c11" => true,
        "$(node.name)₊c12" => false,
        "$(node.name)₊c13" => false,
        "$(node.name)₊c14" => true,
        "$(node.name)₊c21" => true,
        "$(node.name)₊c22" => false,
        "$(node.name)₊c23" => false,
        "$(node.name)₊c24" => false
    ) =#
    return pops, node
end

function JansenRit(node::Node)

    pops = [Population(1, "PYR", node; 
                input_dynamics_spec="second_order_kernel",
                sends_internode_output=true),
            Population(2, "EIN", node; 
                input_dynamics_spec="second_order_kernel",
                sensory_conn_func="linear",
                internode_conn_func="linear",
                ),
            Population(3, "IIN", node; 
                input_dynamics_spec="second_order_kernel")
            ]
    
    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[2, 3, 4, 7]] .= "saturating_sigmoid"
    node.brain_source = "$(node.name)₊x21 - $(node.name)₊x31"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-1.0, 1.0),      # PYR
        "$(node.name)₊x12" => (-220.0, -200.0), # PYR
        "$(node.name)₊x21" => (-5.0, -1.0),     # EIN
        "$(node.name)₊x22" => (-1.0, -0.1),     # EIN
        "$(node.name)₊x31" => (-9.0, -5.0),     # IIN
        "$(node.name)₊x32" => (-260.0, -250.0)  # IIN
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # Pyramidal population (P)
        "$(node.name)₊c11" => (100., ),    # a0: EPSP amplitude (P)
        "$(node.name)₊c12" => (0.0325, ),# A0/a0
        "$(node.name)₊c13" => (1.0, ),       # not used
        "$(node.name)₊c14" => (2*2.5, ),   # e0: max firing rate (P)
        "$(node.name)₊c15" => (0.56, ),      # r: sigmoid slope
        "$(node.name)₊c16" => (6.0, ),       # θ: sigmoid threshold
        "$(node.name)₊c17" => (1.0, ),       # not used
        "$(node.name)₊c18" => (1.0, ),       # not used
        "$(node.name)₊c19" => (-1.0, ),     # not used
        
        # Excitatory interneuron (E)
        "$(node.name)₊c21" => (100.0, ),  # a1: time constant (E)
        "$(node.name)₊c22" => (0.0325, ),# A1/a1: EPSP amplitude (E)
        "$(node.name)₊c23" => (1.0, ),       # not used
        "$(node.name)₊c24" => (2*2.5, ),   # e0 (E) – often tied to P
        "$(node.name)₊c25" => (0.56, ),      # r (E)
        "$(node.name)₊c26" => (6.0, ),       # θ (E)
        "$(node.name)₊c27" => (108.0, ),  # C2: E←P connectivity (inner)
        "$(node.name)₊c28" => (135.0, ), # C1: P→sigmoid gain (outer)
    
        # Inhibitory interneuron (I)
        "$(node.name)₊c31" => (50.0, ),    # b: time constant (I)
        "$(node.name)₊c32" => (0.44, ),    # B/b
        "$(node.name)₊c33" => (1.0, ),       # not used
        "$(node.name)₊c34" => (2*2.5, ),   # e0 (I)
        "$(node.name)₊c35" => (0.56, ),      # r (I)
        "$(node.name)₊c36" => (6.0, ),       # θ (I)
        "$(node.name)₊c37" => (33.75, ),   # C4: I←P connectivity (inner)
        "$(node.name)₊c38" => (33.75, 20.0, 50.0)    # C3: P→I connectivity (outer)
    )

#=     node.build_setts.new_param_tunability = Dict{String, Bool}(
        # Pyramidal
        "$(node.name)₊c11" => true,   # A0: main gain, good to tune
        "$(node.name)₊c12" => false,  # a0: fix around 100 s⁻¹
        "$(node.name)₊c13" => false,  # e0: usually fixed
        "$(node.name)₊c14" => false,  # r
        "$(node.name)₊c15" => false,  # θ
        "$(node.name)₊c16" => true,   # outer connectivity
        "$(node.name)₊c17" => true,   # inner connectivity
        "$(node.name)₊c18" => false,  # inhibitory sign (don't touch)

        # Excitatory interneuron
        "$(node.name)₊c21" => true,   # A1: gain
        "$(node.name)₊c22" => false,  # a1
        "$(node.name)₊c23" => true,   # NI conn strength
        "$(node.name)₊c24" => false,  # e0 (E)
        "$(node.name)₊c25" => false,  # r (E)
        "$(node.name)₊c26" => false,  # θ (E)
        "$(node.name)₊c27" => true,   # C2: inner conn
        "$(node.name)₊c28" => true,   # C1: outer conn

        # Inhibitory interneuron
        "$(node.name)₊c31" => true,   # B: IPSP gain
        "$(node.name)₊c32" => false,  # b
        "$(node.name)₊c33" => false,  # e0 (I)
        "$(node.name)₊c34" => false,  # r (I)
        "$(node.name)₊c35" => false,  # θ (I)
        "$(node.name)₊c36" => true,   # C3: inner conn
        "$(node.name)₊c37" => true    # C4: outer conn
    )
 =#
    return pops, node
end

function Wendling(node::Node)

    pops = [
        Population(1, "PYR", node; 
                        input_dynamics_spec="second_order_kernel", 
                        sends_internode_output=true),
        Population(2, "EIN", node; 
                        input_dynamics_spec="second_order_kernel",
                        sensory_conn_func="linear",
                        internode_conn_func="linear"),
        Population(3, "IINS", node; input_dynamics_spec="second_order_kernel"),
        Population(4, "IINF1", node; input_dynamics_spec="second_order_kernel"),
        Population(5, "IINF2", node; input_dynamics_spec="second_order_kernel")
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[2, 3, 4, 5, 6, 11, 16, 24]] .= "saturating_sigmoid"
    node.brain_source = "$(node.name)₊x21 - $(node.name)₊x31 - $(node.name)₊x41"

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # ── dendrite‑projecting loop (Py kernel) ──────────────────────────
        "$(node.name)₊c11" => (100.0, ),   # a1
        "$(node.name)₊c12" => (0.06,   ),  # A1/a1
        "$(node.name)₊c13" => (1.0,   ),   # not used
        "$(node.name)₊c14" => (2*2.5,   ), # e0
        "$(node.name)₊c15" => (0.56,  ),  # r
        "$(node.name)₊c16" => (6.0,   ),  # θ
        "$(node.name)₊c17" => (1.0,   ),  # not used 
        "$(node.name)₊c18" => (1.0,   ),  # not used 
        "$(node.name)₊c19" => (-1.0,   ),  # not used 
        "$(node.name)₊c110" => (-1.0,   ),  # not used

        # ── excitatory loop (E kernel) ────────────────────────────────────
        "$(node.name)₊c21" => (100.0, ),  # a2
        "$(node.name)₊c22" => (0.06,   ),  # A2
        "$(node.name)₊c23" => (1.0,   ),  # not used
        "$(node.name)₊c24" => (2*2.5,   ),  # e0
        "$(node.name)₊c25" => (0.56,  ),  # r
        "$(node.name)₊c26" => (6.0,   ),  # θ
        "$(node.name)₊c27" => (108.0, ),  # C2
        "$(node.name)₊c28" => (135.0, ),  # C1

        # ── slow inhibition Py ← I_slow ──────────────────────────────────
        "$(node.name)₊c31" => (50.0,  ),  # a3 (=b)
        "$(node.name)₊c32" => (0.8,  ),   # A3 (=B) / a3
        "$(node.name)₊c33" => (1.0,   ),   # not used
        "$(node.name)₊c34" => (2*2.5,   ), # e0
        "$(node.name)₊c35" => (0.56,  ),  # r
        "$(node.name)₊c36" => (6.0,   ),  # θ
        "$(node.name)₊c37" => (33.75, ),  # C3 (= 0.25·C1)
        "$(node.name)₊c38" => (33.75, ),  # C4 (= 0.25·C1)

        # ── fast inhibition Py ← I_fast ──────────────────────────────────
        "$(node.name)₊c41" => (350.0, ),  # a4 (=g)
        "$(node.name)₊c42" => (20.0/350.0,  ),  # A4 (=G) / a4
        "$(node.name)₊c43" => (1.0,   ),   # not used
        "$(node.name)₊c44" => (2*2.5,   ),  # e0
        "$(node.name)₊c45" => (0.56,  ),  # r
        "$(node.name)₊c46" => (6.0,   ),  # θ
        "$(node.name)₊c47" => (108.,  ),  # C5 (= 0.3·C1)
        "$(node.name)₊c48" => (40.5, ),  # C6 (= 0.1·C1)
        "$(node.name)₊c49" => (-13.5, ),  # C7 (= 0.8·C1)

        # ── I_slow → I_fast side loop ────────────────────────────────────
        "$(node.name)₊c51" => (50.0,  ),  # a3 (=b)
        "$(node.name)₊c52" => (0.8,  ),  # A3 (=B) / a3
        "$(node.name)₊c53" => (1.0,   ),   # not used
        "$(node.name)₊c54" => (2*2.5,   ),  # e0
        "$(node.name)₊c55" => (0.56,  ),  # r
        "$(node.name)₊c56" => (6.0,   ),  # θ
        "$(node.name)₊c57" => (1., ),  #
        "$(node.name)₊c58" => (33.75,   )   #C3 unity (not used)
    )

    return pops, node
end

function MoranDavidFriston(node::Node)
    pops = [Population(1, "PYR", node;
                        input_dynamics_spec=["second_order_kernel", "second_order_kernel"],
                        output_dynamics_spec="difference",
                        sends_internode_output=true),                  
            Population(2, "IIN", node; 
                        input_dynamics_spec=["second_order_kernel", "second_order_kernel"],
                        output_dynamics_spec="difference"),
            Population(3, "EIN", node; 
                        input_dynamics_spec="second_order_kernel",
                        sensory_conn_func="linear",
                        internode_conn_func="linear")
    ]
    
    # Build a pop-conn matrix as in original
    node.n_pops = length(pops)
    pop_conn_size = sum([length(pop.build_setts.input_dynamics_spec) for pop in pops])
    node.build_setts.pop_conn = fill("none", pop_conn_size * pop_conn_size)
    node.build_setts.pop_conn[[3, 5, 8, 10, 12, 14, 17, 19, 21]] .= "baseline_sigmoid"
    node.brain_source = "$(node.name)₊x14"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 0.0),
        "$(node.name)₊x12" => (0.0, 0.0),
        "$(node.name)₊x13" => (0.0, 0.0),
        "$(node.name)₊x14" => (0.0, 0.0),
        "$(node.name)₊x15" => (0.0, 0.0),
        "$(node.name)₊x21" => (0.0, 0.0),
        "$(node.name)₊x22" => (0.0, 0.0),
        "$(node.name)₊x23" => (0.0, 0.0),
        "$(node.name)₊x24" => (0.0, 0.0),
        "$(node.name)₊x25" => (0.0, 0.0),
        "$(node.name)₊x31" => (0.0, 0.0),
        "$(node.name)₊x32" => (0.0, 0.0),
    )   

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # ─────────────────────────  Recurrent excitatory loop (PYR)  ─────────────────────────
        "$(node.name)₊c11" => (250.0, ),   # κₑ  (excitatory inverse time-const.)
        "$(node.name)₊c12" => (10.0/250., ),      # He /  κₑ  (EPSP amplitude) 
        "$(node.name)₊c13" => (1.0, ),         # not used
        "$(node.name)₊c14" => (62.5, ),      # κᵢ
        "$(node.name)₊c15" => (22.0/62.5, ),      # Hi /  κi
        "$(node.name)₊c16" => (1.0, ),         # not used
        "$(node.name)₊c17" => (2.0, ),         # ρ₁
        "$(node.name)₊c18" => (1.0, ),         # ρ₂
        "$(node.name)₊c19" => (64.0, ),      # γ₄
        "$(node.name)₊c110" => (1.0, ),         # not used
        "$(node.name)₊c111" => (2.0, ),         # ρ₁  (sigmoid slope)  
        "$(node.name)₊c112" => (1.0, ),         # ρ₂  (sigmoid thresh., inhib.)
        "$(node.name)₊c113" => (128.0, ),   # γ₂  (gain inside sigmoid term)
        "$(node.name)₊c114" => (1.0, ),         # not used

        # ───────────────────────── Inhibitory  ──────────────────────
        "$(node.name)₊c21" => (250.0, ),   # κₑ
        "$(node.name)₊c22" => (10.0/250., ),      # He  /  κₑ
        "$(node.name)₊c23" => (1.0, ),         # not used
        "$(node.name)₊c24" => (62.5, ),      # κᵢ
        "$(node.name)₊c25" => (22.0/62.5, ),      # Hi
        "$(node.name)₊c26" => (1.0, ),         # not used
        "$(node.name)₊c27" => (2.0, ),         # ρ₁
        "$(node.name)₊c28" => (1.0, ),         # ρ₂
        "$(node.name)₊c29" => (1.0, ),         # γ₅
        "$(node.name)₊c210" => (1.0, ),         # not used
        "$(node.name)₊c211" => (2.0, ),         # ρ₁
        "$(node.name)₊c212" => (1.0, ),        # ρ₂
        "$(node.name)₊c213" => (64.0, ),       # γ₃  
        "$(node.name)₊c214" => (1.0, ),         # not used

        # ─────────────────────────  Excitatory interneuron loop (EI)  ────────────────────────
        "$(node.name)₊c31" => (250.0, ),   # κₑ  (same τ as PYR loop)
        "$(node.name)₊c32" => (10.0/250.0, ),      # He / κₑ  (again, for EI loop)
        "$(node.name)₊c33" => (1.0, ),         # not used
        "$(node.name)₊c34" => (2.0, ),         # ρ₁  (sigmoid slope, EI)
        "$(node.name)₊c35" => (1.0, ),         # ρ₂  (sigmoid thresh., EI)
        "$(node.name)₊c36" => (128.0, ),   # γ₁  (gain for EI drive)
        "$(node.name)₊c37" => (1.0, ),         # not used

    )

    return pops, node
end

function LileyWright(node::Node)
    pops = [
        Population(1, "E", node; 
                        input_dynamics_spec=["second_order_kernel", "second_order_kernel"], 
                        output_dynamics_spec="membrane_integrator",
                        input2output_conn_func="linear",
                        sensory_conn_func=["linear", "none"],
                        internode_conn_func=["linear", "none"],
                        noise_dynamics_spec=["stochastic", ""],
                        noise_dynamics="c",
                        sends_internode_output=true),
        Population(2, "I", node; 
                        input_dynamics_spec=["second_order_kernel", "second_order_kernel"], 
                        output_dynamics_spec="membrane_integrator",
                        sensory_conn_func=["none", "linear"],
                        input2output_conn_func="linear")
    ]
    node.n_pops = length(pops)
    pop_conn_size = sum([length(pop.build_setts.input_dynamics_spec) for pop in pops])
    node.build_setts.pop_conn = fill("none", pop_conn_size * pop_conn_size)
    node.build_setts.pop_conn[[1, 8, 10, 11]] .= "saturating_sigmoid"
    node.brain_source = "$(node.name)₊x15"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (55.0, 56.0),  # E
        "$(node.name)₊x12" => (0.0,   0.0),
        "$(node.name)₊x13" => (25.0, 26.0),
        "$(node.name)₊x14" => (0.0,   0.0), 
        "$(node.name)₊x15" => (-70.0, -70.0),
        "$(node.name)₊x21" => (25.0, 26.0),  # I
        "$(node.name)₊x22" => (0.0,   0.0),
        "$(node.name)₊x23" => (66.0, 67.0),
        "$(node.name)₊x24" => (0.0,   0.0),
        "$(node.name)₊x25" => (-70.0, -70.0)
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (450.,),              # γ_ee, PSP decay rate (seconds⁻¹) - tuned for 10Hz
        "$(node.name)₊c12" => (ℯ*0.71/450.,),       # e * Γ_ee / γ_ee, PSP peak amplitude, e→e (unit: mV)
        "$(node.name)₊c13" => (1.,),                # not used
        "$(node.name)₊c14" => (97.5,),              # γ_ie, PSP decay rate (seconds⁻¹) - tuned for 10Hz
        "$(node.name)₊c15" => (ℯ*0.71/97.5,),       # e * Γ_ie / γ_ie, PSP peak amplitude, i→e (unit: mV)
        "$(node.name)₊c16" => (1.,),                # not used
        "$(node.name)₊c17" => (0.0282,),            # τ_e, PSP time constant (seconds) - tuned for 10Hz
        "$(node.name)₊c18" => (-70.0,),             # h_e_r, resting potential (unit: mV)
        "$(node.name)₊c19" => (45.0,),              # h_ee_eq, # equilibrium potential, e→e (unit: mV)
        "$(node.name)₊c110" => (-90.0,),            # h_ie_eq, # equilibrium potential, i→e (unit: mV)
        "$(node.name)₊c111" => (500.,),             # S_i_max (e0 = Smax)
        "$(node.name)₊c112" => (sqrt(2)/5.,),       # σ_i ... r = sqrt(2)/σ
        "$(node.name)₊c113" => (-50.,),             # μ_i
        "$(node.name)₊c114" => (500.,),             # N_ie
        "$(node.name)₊c115" => (1.,),               # not used
        "$(node.name)₊c116" => (500.,),             # S_e_max
        "$(node.name)₊c117" => (sqrt(2)/5.,),       # σ_e
        "$(node.name)₊c118" => (-50.,),             # μₑ   
        "$(node.name)₊c119" => (3000.,),            # N_ee_b   
        "$(node.name)₊c120" => (1.,),               # not used
        "$(node.name)₊c121" => (0.3*1e6*0.71*ℯ,),   # noise amplitude scaling (stochastic for I_ee = γ_ee * P.Γ_ee * P.p_ee_sd)

        "$(node.name)₊c21" => (97.5,),              # γ_ii, PSP decay rate (seconds⁻¹) - tuned for 10Hz
        "$(node.name)₊c22" => (ℯ*0.71/97.5,),       # e * Γ_ii / γ_ii, PSP peak amplitude, i→i (unit: mV)
        "$(node.name)₊c23" => (1.,),                # not used
        "$(node.name)₊c24" => (450.,),              # γ_ei , PSP decay rate (seconds⁻¹) - tuned for 10Hz
        "$(node.name)₊c25" => (ℯ*0.71/450.,),       # e * Γ_ei / γ_ei, PSP peak amplitude, e→i (unit: mV)
        "$(node.name)₊c26" => (1.,),                # not used
        "$(node.name)₊c27" => (0.0126,),            # τ_i, PSP time constant (seconds) - tuned for 10Hz
        "$(node.name)₊c28" => (-70.,),              # h_i_r, resting potential (unit: mV)
        "$(node.name)₊c29" => (-90.,),              # h_ii_eq, equilibrium potential, i→i (unit: mV)
        "$(node.name)₊c210" => (45.,),              # h_ei_eq equilibrium potential, e→i (unit: mV)
        "$(node.name)₊c211" => (500.,),             # S_e_max
        "$(node.name)₊c212" => (sqrt(2)/5.,),       # σ_e
        "$(node.name)₊c213" => (-50.,),             # μₑ    
        "$(node.name)₊c214" => (3000.,),            # N_ei
        "$(node.name)₊c215" => (1.,),               # not used
        "$(node.name)₊c216" => (500,),              # S_i_max
        "$(node.name)₊c217" => (sqrt(2)/5.,),       # σ_i
        "$(node.name)₊c218" => (-50.,),             # μ_i
        "$(node.name)₊c219" => (500.,),             # N_ii
        "$(node.name)₊c220" => (1.,),               # not used
   ) 
    return pops, node
end

function RobinsonRennieWrightCortex(node::Node)
    pops = [
        Population(1, "Ve", node; 
                            input_dynamics_spec="second_order_kernel", 
                            output_dynamics_spec="spatial_gradient", 
                            internode_conn_func="linear",
                            sends_internode_output=true,
                            ),
        Population(2, "Vi", node; input_dynamics_spec="second_order_kernel",
                            internode_conn_func="linear"
                            ),
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[1, 2]] .= "linear" 
    node.build_setts.pop_conn[[3, 4]] .= "saturating_sigmoid" 
    node.brain_source = "$(node.name)₊x13"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0006344, 0.0006344),  # Ve
        "$(node.name)₊x12" => (0.0, 0.0),  # Ve
        "$(node.name)₊x13" => (3.175, 3.175),  # Φe
        "$(node.name)₊x14" => (0.0, 0.0),  # Φe      
        "$(node.name)₊x21" => (0.0006344, 0.0006344),  # Vi
        "$(node.name)₊x22" => (0.0, 0.0)   # Vi
    )

    # Original RRWC params (from your script)
    α   = 83.33
    β   = 4 * 83.33
    γ   = 116.0
    τ   = 80e-3 / 2

    Qmax = 340.0
    θ    = 12.92e-3
    σ    = 3.8e-3

    nu_ee = 3.03e-3
    nu_ei = -6.0e-3
    nu_ie = 3.03e-3
    nu_ii = -6.0e-3

    # Kernel reparam (matches your ENMEEG form exactly)
    ω = sqrt(α * β)                 # = 166.66
    ζ = (α + β) / (2 * ω)           # = 1.25

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (ω, ),            # ω = sqrt(α * β)
        "$(node.name)₊c12" => (1., ),           # not used
        "$(node.name)₊c13" => (ζ, ),            # ζ = (α + β) / (2 * w)
        "$(node.name)₊c14" => (Qmax, ),        # Qmax
        "$(node.name)₊c15" => (1/σ, ),     # 1/σ
        "$(node.name)₊c16" => (θ, ),     # θ
        "$(node.name)₊c17" => (1., ),           # not used
        "$(node.name)₊c18" => (1., ),           # not used 
        "$(node.name)₊c19" => (γ, ),        # γ 
        "$(node.name)₊c110" => (nu_ee, ),     # nu_ee
        "$(node.name)₊c111" => (Qmax, ),       # Qmax
        "$(node.name)₊c112" => (1/σ, ),    # σ
        "$(node.name)₊c113" => (θ, ),    # θ
        "$(node.name)₊c114" => (nu_ei, ),     # nu_ei
        "$(node.name)₊c115" => (1.0, ),         # not used
#=         "$(node.name)₊c116" => (340.0, ),       # Qmax
        "$(node.name)₊c117" => (1/3.8e-3, ),    # σ
        "$(node.name)₊c118" => (θ, ),    # θ
        "$(node.name)₊c119" => (1., ),          # not used
        "$(node.name)₊c120" => (2.06e-3, ),     # nu_es =#
  
        "$(node.name)₊c21" => (ω, ),           
        "$(node.name)₊c22" => (1., ),     
        "$(node.name)₊c23" => (ζ, ),
        "$(node.name)₊c24" => (nu_ie, ),      # nu_ie
        "$(node.name)₊c25" => (Qmax, ),        # Qmax
        "$(node.name)₊c26" => (1/σ, ),     # σ
        "$(node.name)₊c27" => (θ, ),     # θ
        "$(node.name)₊c28" => (nu_ii, ),      # nu_ii
        "$(node.name)₊c29" => (1., ),           # not used
#=         "$(node.name)₊c210" => (Qmax, ),       # Qmax
        "$(node.name)₊c211" => (1/σ, ),    # σ
        "$(node.name)₊c212" => (θ, ),    # θ
        "$(node.name)₊c213" => (2.06e-3, ),     # nu_is
        "$(node.name)₊c214" => (1., ),          # not used =#
        
        "$(node.name)₊τ_x13" => (τ, ), #
    )
    return pops, node
end

function RobinsonRennieWrightThalamus(node::Node)
    pops = [
        Population(1, "Vs", node; 
                            input_dynamics_spec="second_order_kernel", 
                            sensory_conn_func="linear",
                            internode_conn_func="linear",
                            sends_internode_output=true),
        Population(2, "Vr", node; 
                            input_dynamics_spec="second_order_kernel",
                            internode_conn_func="linear")
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[2, 3]] .= "saturating_sigmoid"
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-0.003234, -0.003234),  # Vs
        "$(node.name)₊x12" => (0.0, 0.0),  # Vs
        "$(node.name)₊x21" => (0.005676, 0.005676),  # Vr
        "$(node.name)₊x22" => (0.0, 0.0)   # Vr
    )

    α = 83.33
    β = α * 4
    ω = sqrt(α * β)
    ζ = (α + β) / (2 * ω)

    Qmax = 340.0
    θ    = 12.92e-3
    σ    = 3.8e-3
    τ   = 80e-3 / 2

    nu_sr = -0.83e-3
    nu_rs = 0.03e-3

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (ω, ),             # ω = sqrt(α * β)
        "$(node.name)₊c12" => (1., ),            # not used
        "$(node.name)₊c13" => (ζ, ),             # ζ = (α + β) / (2 * w)
        "$(node.name)₊c14" => (Qmax, ),        # Qmax
        "$(node.name)₊c15" => (1/σ, ),     # σ
        "$(node.name)₊c16" => (θ, ),     # θ
        "$(node.name)₊c17" => (nu_sr, ),     # nu_sr
        "$(node.name)₊c18" => (1., ),           # not used
        # "$(node.name)₊c19" => (2.18e-3, ),      # nu_se

        "$(node.name)₊c21" => (ω, ),           
        "$(node.name)₊c22" => (1., ),
        "$(node.name)₊c23" => (ζ, ),
        "$(node.name)₊c24" => (Qmax, ),        # Qmax
        "$(node.name)₊c25" => (1/σ, ),     # σ
        "$(node.name)₊c26" => (θ, ),     # θ
        "$(node.name)₊c27" => (nu_rs, ),      # nu_rs
        "$(node.name)₊c28" => (1., ),           # not used
        # "$(node.name)₊c29" => (0.33e-3, ),      # nu_re

        # "$(node.name)₊τ_x11" => (τ, ), # τ_s
    )

    return pops, node
end    

function WongWang(node::Node)

    pops = [Population(1, "E", node;
                      input_dynamics_spec="gating_kinetics",
                      sensory_conn_func="relaxed_rectifier",
                      internode_conn_func="relaxed_rectifier",
                      noise_dynamics_spec="stochastic",
                      noise_dynamics="c",
                      sends_internode_output=true),
            Population(2, "I", node;
                      input_dynamics_spec="exp_kernel",
                      sensory_conn_func="relaxed_rectifier",
                      internode_conn_func="relaxed_rectifier",
                      noise_dynamics_spec="stochastic",
                      noise_dynamics="c")
            ]

    # Each node has only one population (E), and receives from itself and external inputs
    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("relaxed_rectifier", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    # Initial condition for synaptic gating variable S(t)
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 1.0),
        "$(node.name)₊x21" => (0.0, 1.0)
    )
    
    # Parameters: (default, min, max)
    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (0.03, ),       # τ_E: NMDA decay (optimized for 10Hz: was 0.1)
        "$(node.name)₊c12" => (0.5128, ),     # γ_E: gain (optimized for 10Hz: was 0.641)
        "$(node.name)₊c13" => (310.0, ),      # a: input gain
        "$(node.name)₊c14" => (125.0, ),      # b: threshold
        "$(node.name)₊c15" => (0.16, ),       # d: steepness
        "$(node.name)₊c16" => (0.21, ),       # w*JN: recurrent weight (0.9*0.2348)
        "$(node.name)₊c17" => (-1.0, ),       # Ji:  inhibitory coupling
        "$(node.name)₊c18" => (0.02, ),       # stochastic scaling

        "$(node.name)₊c21" => (0.003, ),      # τ_I: (optimized for 10Hz: was 1.0)
        "$(node.name)₊c22" => (100.0, ),      # -1/τ = -c22/c21
        "$(node.name)₊c23" => (615.0, ),      # a: input gain
        "$(node.name)₊c24" => (177.0, ),      # b: threshold
        "$(node.name)₊c25" => (0.087, ),      # d: steepness
        "$(node.name)₊c26" => (0.15, ),       # w*JI: recurrent weight
        "$(node.name)₊c27" => (-1.0, ),       # Je: excitatory coupling
        "$(node.name)₊c28" => (0.02, )        # stochastic scaling
    )

#= 
    # Tunability settings
    node.build_setts.new_param_tunability = Dict{String, Bool}(
        "$(node.name)₊c11" => true,   # τs
        "$(node.name)₊c12" => true,   # γ
        "$(node.name)₊c13" => false,  # a (fixed)
        "$(node.name)₊c14" => false,  # b (fixed)
        "$(node.name)₊c15" => false,  # d (fixed)
        "$(node.name)₊c16" => true,   # JN (can tune recurrence)
        "$(node.name)₊c17" => false,   # scaling usually fixed at 1
        "$(node.name)₊c21" => true,   # τs
        "$(node.name)₊c22" => true,   # γ
        "$(node.name)₊c23" => false,  # a (fixed)
        "$(node.name)₊c24" => false,  # b (fixed)
        "$(node.name)₊c25" => false,  # d (fixed)
        "$(node.name)₊c26" => true,   # JI (can tune recurrence)
        "$(node.name)₊c27" => false    # scaling usually fixed at 1
    ) =#
    return pops, node
end

function WongWangReduced(node::Node)

    pops = [Population(1, "E", node;
                      input_dynamics_spec="gating_kinetics",
                      sensory_conn_func="relaxed_rectifier",
                      internode_conn_func="relaxed_rectifier",
                      noise_dynamics_spec="stochastic",
                      noise_dynamics="c",
                      sends_internode_output=true)]

    # Each node has only one population (E), and receives from itself and external inputs
    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("relaxed_rectifier", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    # Initial condition for synaptic gating variable S(t)
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 1.0)
    )
    
    # Parameters: (default, min, max)
    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (0.04, ),       # τ_s: NMDA decay (optimized for 10Hz: was 0.1)
        "$(node.name)₊c12" => (0.5128, ),     # γ: gain (optimized for 10Hz: was 0.641)
        "$(node.name)₊c13" => (270.0, ),      # a: input gain
        "$(node.name)₊c14" => (108.0, ),      # b: threshold
        "$(node.name)₊c15" => (0.154, ),      # d: steepness
        "$(node.name)₊c16" => (0.235, ),      # w*JN: recurrent weight ( 0.9 * 0.2609)
        "$(node.name)₊c17" => (0.02, ),       # stochastic noise amplitude scaling
    )


#=     # Tunability settings
    node.build_setts.new_param_tunability = Dict{String, Bool}(
        "$(node.name)₊c11" => true,   # τs
        "$(node.name)₊c12" => true,   # γ
        "$(node.name)₊c13" => false,  # a (fixed)
        "$(node.name)₊c14" => false,  # b (fixed)
        "$(node.name)₊c15" => false,  # d (fixed)
        "$(node.name)₊c16" => true,   # JN (can tune recurrence)
        "$(node.name)₊c17" => false   # scaling usually fixed at 1
    ) =#
    return pops, node
end

function LarterBreakspear(node::Node)

    pops = [
            Population(1, "LB", node; 
                        input_dynamics_spec="voltage_gated_dynamics",
                        internode_conn_func="linear",
                        sensory_conn_func="linear",
                        sends_internode_output=true),
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"  # V population is the brain source

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 0.0),  # E
        "$(node.name)₊x12" => (0.0, 0.0),  
        "$(node.name)₊x13" => (0.0, 0.0),  # I
        )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        # conductances
        "$(node.name)₊c11"  => (1.1,  ),   # gCa
        "$(node.name)₊c12"  => (2.0,  ),   # gK
        "$(node.name)₊c13"  => (0.5,  ),   # gL
        "$(node.name)₊c14"  => (6.7,  ),   # gNa

        # reversal potentials
        "$(node.name)₊c15"  => (1.0,  ),   # VCa
        "$(node.name)₊c16"  => (-0.7, ),   # VK
        "$(node.name)₊c17"  => (-0.5, ),   # VL
        "$(node.name)₊c18"  => (0.53, ),   # VNa

        # channel thresholds & widths
        "$(node.name)₊c19"  => (-0.01, ),  # TCa
        "$(node.name)₊c110" => (0.15,  ),  # d_Ca
        "$(node.name)₊c111" => (0.3,   ),  # TNa
        "$(node.name)₊c112" => (0.15,  ),  # d_Na
        "$(node.name)₊c113" => (0.0,   ),  # TK
        "$(node.name)₊c114" => (0.3,   ),  # d_K

        # population-gain sigmoids
        "$(node.name)₊c115" => (0.0,   ),  # VT
        "$(node.name)₊c116" => (0.65,  ),  # d_V
        "$(node.name)₊c117" => (0.0,   ),  # ZT
        "$(node.name)₊c118" => (0.7,   ),  # d_Z

        # synaptic strengths
        "$(node.name)₊c119" => (0.4,   ),  # aee
        "$(node.name)₊c120" => (2.0,   ),  # aei
        "$(node.name)₊c121" => (2.0,   ),  # aie
        "$(node.name)₊c122" => (1.0,   ),  # ane
        "$(node.name)₊c123" => (0.4,   ),  # ani

        # NMDA / coupling
        "$(node.name)₊c124" => (0.1,   ),  # C
        "$(node.name)₊c125" => (0.25,  ),  # rNMDA

        # time constants / rates
        "$(node.name)₊c126" => (0.7,   ),  # phi
        "$(node.name)₊c127" => (1.0,   ),  # tau_K
        "$(node.name)₊c128" => (0.1,   ),  # b
    )

    return pops, node
end

function HarmonicOscillator(node::Node)
    pops = [Population(1, "HO", node; 
                        input_dynamics_spec="second_order_kernel", 
                        sensory_conn_func="linear",
                        internode_conn_func="linear",
                        sends_internode_output=true)]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-1., 1.),  # Vs
        "$(node.name)₊x12" => (-1.0, 1.0),  # Vs
    )
    η = 0.05  # damping coefficient
    ω = 2*π*10.0  # natural frequency (rad/s)
    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (ω,), # (1/((2*π*10.0)^2),),
        "$(node.name)₊c12" => (1/((2*π*10.0)^2), ),
        "$(node.name)₊c13" => (η/ω,), # (ω,)
    )

    return pops, node

end

function Montbrio(node::Node)
    # One Montbrio population with 2 state variables: x1 = R, x2 = V
    # eq1:  dR/dt = c11 + c16*SI(t) + c13*R*V
    # eq2:  dV/dt = c12 + c17*SI(t) + c14*V^2 + c15*R^2    
    pops = [
        Population(1, "MBR", node;

            input_dynamics_spec = "x1*x2, x1 + x1*x1 + x2*x2",
            sensory_conn_func   = "linear", 
            internode_conn_func = "linear",
            sends_internode_output = true
        )
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)

    node.brain_source = "$(node.name)₊x11"   # i.e. R(t)

    # --- Initial conditions --------------------------------------------------
    # Typical: start near a low-rate asynchronous state
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.5, 0.5),   # R(0)
        "$(node.name)₊x12" => (0.0, 0.0),   # V(0) 
    )

    # --- Parameters ----------------------------------------------------------
    τ   = 0.1   # seconds

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (1/(π*τ^2), ),   # constant drive to R
        "$(node.name)₊c12" => (-1.6/τ, ),   # constant drive to V
        "$(node.name)₊c13" => (2.0/τ,  ), #
        "$(node.name)₊c14" => (15.5,  ),
        "$(node.name)₊c15" => (-τ*π^2, ),
        "$(node.name)₊c16" => (1/τ, ),   #
        "$(node.name)₊c17" => (0.0, ),    # SI1
        "$(node.name)₊c18" => (1.0, )    # SI2
    )

    return pops, node
end

function FitzHughNagumo(node::Node)
    pops = [Population(1, "FHN", node; 
                        input_dynamics_spec="x1 + x1*x1*x1 + x2, x1 + x2",
                        sensory_conn_func="linear",
                        internode_conn_func="linear",
                        sends_internode_output=true)]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-2.0, 2.0),  # x1
        "$(node.name)₊x12" => (-2.0, 2.0)   # x2
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (0.0, ),      # for const term
        "$(node.name)₊c12" => (0.28, ),   # for const term: a / τ
        "$(node.name)₊c13" => (1.0, ),      # dx: x1
        "$(node.name)₊c14" => (-0.33, ),    # dx: for x1^3
        "$(node.name)₊c15" => (-1.0, ),   # dx: x2
        "$(node.name)₊c16" => (0.4, ),      # 1 / τ
        "$(node.name)₊c17" => (-0.32, ),# - b / τ
        "$(node.name)₊c18" => (1.0, ),      # for SI1
        "$(node.name)₊c19" => (0.0, ),      # for SI2
    )
    return pops, node
end

function VanDerPol(node::Node)
    pops = [Population(1, "VDP", node; 
                        input_dynamics_spec="x2, x1 + x2 + x1*x1*x2", 
                        sensory_conn_func="linear",
                        internode_conn_func="linear",
                        sends_internode_output=true)
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 0.0),  # x1
        "$(node.name)₊x12" => (1.0, 1.0)   # x2
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (0.0, ),      # for const term
        "$(node.name)₊c12" => (0.0, ),      # for const term
        "$(node.name)₊c13" => (1.0, ),      # for dx: x2
        "$(node.name)₊c14" => (-1.0, ),   # for dy: x1
        "$(node.name)₊c15" => (1.0, ),      # for dy: x^2*y (μ)
        "$(node.name)₊c16" => (-1.0, ),   # for dy: x2
        "$(node.name)₊c17" => (1.0, ),      # for input term (I(t))
    )

    node.brain_source = "$(node.name)₊x11"
    return pops, node
end

function StuartLandau(node::Node)

    pops = [Population(1, "SL", node;
                      input_dynamics_spec="x1 + x2 + x1*x1*x1 + x1*x2*x2, x1 + x2 + x2*x2*x2 + x1*x1*x2",
                      sensory_conn_func="linear",
                      internode_conn_func="linear",
                      sends_internode_output=true)
                      ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (2.0, 2.0),    # initial x
        "$(node.name)₊x12" => (0.0, 0.0)     # initial y
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (0.0, ),     # constant term
        "$(node.name)₊c12" => (0.0, ),     # constant term
        "$(node.name)₊c13" => (1.0, ),     # dx: x1 term (μ > 0.0)
        "$(node.name)₊c14" => (-10.0, ), # dx: x2
        "$(node.name)₊c15"=>  (-1.0, ),  # dx: x3
        "$(node.name)₊c16" => (-1.0, ), # dx: x1*x2²
        "$(node.name)₊c17" => (10., ),    # dy: x1
        "$(node.name)₊c18" => (1.0, ),    # dy: x2
        "$(node.name)₊c19" => (-1.0, ), # dy: x2*x2*x2
        "$(node.name)₊c110" => (-1.0, ), # dy: x1²*x2
        "$(node.name)₊c111" => (1.0, ),   # for input term (I(t)
    )

    return pops, node
end

function DuffingOscillator(node::Node)
    #du[1] = x2
    #du[2] = δ*x2 + α*x1 + β*x1^3 + γ*I, where I = (cos(ω*t)) 

    pops = [Population(1, "DO", node;
                        input_dynamics_spec="x2, x1 + x2 + x1*x1*x1",
                        sensory_conn_func="linear",
                        internode_conn_func="linear",
                        sends_internode_output=true)]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (0.0, 0.0),  # x
        "$(node.name)₊x12" => (1.0, 1.0)   # y
    )

    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
    "$(node.name)₊c11" => (0., ),         # for const term
    "$(node.name)₊c12" => (0.0, ),      # for const term
    "$(node.name)₊c13" => (1.0, ),      # for x2
    "$(node.name)₊c14" => (100., ),   # for x2
    "$(node.name)₊c15" => (-0.05, ),# for x1
    "$(node.name)₊c16" => (-200., ),# for x1^3
    "$(node.name)₊c17" => (1., ),       # for input term (I(t)
    )
    return pops, node
end

function PhaseDynamics(node::Node)
    pops = [Population(1, "PD", node;
                        input_dynamics_spec="linear_kernel",
                        sensory_conn_func="fourier_basis",
                        internode_conn_func="fourier_basis",
                        sends_internode_output=true)]
    node.n_pops = length(pops)
    node.brain_source = "$(node.name)₊x11"

    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-1.5, 1.5),  # phi
    )
    node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
        "$(node.name)₊c11" => (2.0*π*10.0, ),  # intrinsic frequency
        "$(node.name)₊c12" => (1.0, ),           # input scaling
    )
    return pops, node   
end

function EpileptorGeneral(node::Node)

    pops = [
        Population(1, "P1", node;
            input_dynamics_spec="slowfast_piecewise_poly_kernel",
            sensory_conn_func="linear",
            sends_internode_output=true),

        Population(2, "P2", node;
            input_dynamics_spec="slowfast_piecewise_poly_kernel",
            internode_conn_func="linear",
            noise_dynamics_spec="additive",
            noise_dynamics="rand(Normal(0, 0.00025))"),

        Population(3, "RS", node;
            input_dynamics_spec="x2 + x1*x1 + x1*x1*x1, x1 + x2",
            internode_conn_func="linear",
            noise_dynamics_spec="additive",
            noise_dynamics="rand(Normal(0, 0.02))"),
    ]

    node.n_pops = length(pops)

    # pop_conn: you know best which (i,j) these indices correspond to;
    # keeping your settings here.
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    node.build_setts.pop_conn[[2]] .= "linear"
    node.build_setts.pop_conn[[4]] .= "piecewise_linear"

    # LFP mixture: 0.1*x2 - 0.1*x1 + 0.9*x_rs
    node.brain_source =
        "0.1 * $(node.name)₊x21 - 0.1 * $(node.name)₊x11 + 0.9 * $(node.name)₊x31"

    # --- initial conditions -------------------------------------------------
    # Epileptor-RS: (x1,y1,z, x2,y2,g, x_rs,y_rs)
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-1.8, -1.4),   # x1
        "$(node.name)₊x12" => (-15.0, -10.0), # y1
        "$(node.name)₊x13" => (3.6, 4.0),     # z

        "$(node.name)₊x21" => (-1.1, -0.9),   # x2
        "$(node.name)₊x22" => (0.001, 0.01),  # y2
        "$(node.name)₊x23" => (0.0, 0.1),     # g   (← was x14 before; this is the slow var of P2)

        "$(node.name)₊x31" => (-2.0, 4.0),    # x_rs
        "$(node.name)₊x32" => (-6.0, 6.0),    # y_rs
    )

    let name = node.name
        # === base epileptor params (matching epileptor_rs_full!) ============
        a      = 1.0
        b      = 3.0
        c1     = 1.0
        d1     = 5.0
        r      = 3.5e-4
        s      = 4.0
        x0     = -1.6
        I1     = 3.1
        m      = 0.0

        aa     = 6.0
        bb     = 2.0
        I2     = 0.45
        tau    = 10.0

        d_rs   = 0.02
        tau_rs = 1.0

        # --------------------------------------------------------------------
        # Pop 1: (x1,y1,z) as slowfast_piecewise_poly_kernel
        # --------------------------------------------------------------------
        pop1_params = Dict{String, Tuple{Vararg{Float64}}}(
            # f_fast(x1,z) reproduces f1(x1,z):
            #   x1<0:  a*x1^3 - b*x1^2
            #   x1>=0: (-m - 0.6(z-4)^2)*x1 = (-m-9.6 + 4.8 z -0.6 z^2)*x1
            "$(name)₊c11"  => (0.0,      ),   # θ_fast = 0
            "$(name)₊c12"  => (0.0,      ),   # a0
            "$(name)₊c13"  => (0.0,      ),   # a1
            "$(name)₊c14"  => (-b,       ),   # a2 = -b
            "$(name)₊c15"  => (a,        ),   # a3 = a
            "$(name)₊c16"  => (-m - 9.6, ),   # b0
            "$(name)₊c17"  => (4.8,      ),   # b1
            "$(name)₊c18"  => (-0.6,     ),   # b2

            # f_slow ≡ 0
            "$(name)₊c19"  => (0.0,      ),   # θ_slow
            "$(name)₊c110" => (0.0,      ),
            "$(name)₊c111" => (0.0,      ),
            "$(name)₊c112" => (0.0,      ),
            "$(name)₊c113" => (0.0,      ),
            "$(name)₊c114" => (0.0,      ),
            "$(name)₊c115" => (0.0,      ),
            "$(name)₊c116" => (0.0,      ),
            "$(name)₊c117" => (0.0,      ),

            # dX1 = y1 - z + I1 - f1(x1,z)
            "$(name)₊c118" => (1.0,      ),   # A1 * Y
            "$(name)₊c119" => (-1.0,     ),   # A2 * W (W=z)
            "$(name)₊c120" => (0.0,      ),   # A3 * U (unused)
            "$(name)₊c121" => (I1,       ),   # A4
            "$(name)₊c122" => (0.0,      ),   # A5 * X
            "$(name)₊c123" => (0.0,      ),   # A6 * X^3

            # dY1 = c1 - d1*x1^2 - y1
            "$(name)₊c124" => (-1.0,     ),   # B1 * Y
            "$(name)₊c125" => (c1,       ),   # B2
            "$(name)₊c126" => (0.0,      ),   # B3 * X
            "$(name)₊c127" => (-d1,      ),   # B4 * X^2

            # dZ = r*(s*(x1 - x0) - z)
            "$(name)₊c128" => (r,        ),   # C1
            "$(name)₊c129" => (s,        ),   # C2
            "$(name)₊c130" => (x0,       ),   # C3
            "$(name)₊c131" => (0.0,      ),   # C4 * U (unused for now)
        )

        # --------------------------------------------------------------------
        # Pop 2: (x2,y2,g) as slowfast_piecewise_poly_kernel
        # --------------------------------------------------------------------
        pop2_params = Dict{String, Tuple{Vararg{Float64}}}(
            # f_fast ≡ 0 for P2 (we encode du4 directly in A* terms)
            "$(name)₊c21"  => (0.0,          ),  # θ_fast
            "$(name)₊c22"  => (0.0,          ),
            "$(name)₊c23"  => (0.0,          ),
            "$(name)₊c24"  => (0.0,          ),
            "$(name)₊c25"  => (0.0,          ),
            "$(name)₊c26"  => (0.0,          ),
            "$(name)₊c27"  => (0.0,          ),
            "$(name)₊c28"  => (0.0,          ),

            # f_slow encodes f2(x2):
            #   x2 < -0.25 : 0
            #   x2 ≥ -0.25: aa*(x2+0.25)
            # and dY2 = (-y2 + f2(x2))/tau
            "$(name)₊c29"  => (-0.25,        ),  # θ_slow
            "$(name)₊c210" => (0.0,          ),  # sL0
            "$(name)₊c211" => (0.0,          ),  # sL1
            "$(name)₊c212" => (0.0,          ),  # sL2
            "$(name)₊c213" => (0.0,          ),  # sL3
            "$(name)₊c214" => (-(aa/tau)*0.25,), # sR0
            "$(name)₊c215" => (-(aa/tau),     ),  # sR1
            "$(name)₊c216" => (0.0,          ),  # sR2
            "$(name)₊c217" => (0.0,          ),  # sR3

            # dX2 = -y2 + x2 - x2^3 + I2 + bb*g - 0.3*(z - 3.5)
            # here W = g, U will carry z (via pop_conn)
            "$(name)₊c218" => (-1.0,         ),  # A1 * Y
            "$(name)₊c219" => (bb,           ),  # A2 * W (g)
            "$(name)₊c220" => (-0.3,         ),  # A3 * U (z)
            "$(name)₊c221" => (I2 + 0.3*3.5, ),  # A4
            "$(name)₊c222" => (1.0,          ),  # A5 * X
            "$(name)₊c223" => (-1.0,         ),  # A6 * X^3

            # dY2 = (-y2 + f2(x2))/tau
            "$(name)₊c224" => (-1.0/tau,     ),  # B1 * Y
            "$(name)₊c225" => (0.0,          ),  # B2
            "$(name)₊c226" => (0.0,          ),  # B3
            "$(name)₊c227" => (0.0,          ),  # B4

            # dW2 ≈ -0.01*g for standalone test
            # (you can later add x1 via U and C4 if you want g to see x1)
            "$(name)₊c228" => (0.01,         ),  # C1
            "$(name)₊c229" => (0.0,          ),  # C2
            "$(name)₊c230" => (0.0,          ),  # C3
            "$(name)₊c231" => (0.0,          ),  # C4 * U
        )

        # --------------------------------------------------------------------
        # RS oscillator (x_rs,y_rs) – cubic oscillator with TVB scaling
        # --------------------------------------------------------------------
        rs_params = Dict{String, Tuple{Vararg{Float64}}}(
            "$(name)₊c31" => (0.0,                 ),  # I_rs
            "$(name)₊c32" => (1.0*d_rs*tau_rs,     ),  # α*y
            "$(name)₊c33" => (3.0*d_rs*tau_rs,     ),  # e*x^2
            "$(name)₊c34" => (-1.0*d_rs*tau_rs,    ),  # f*x^3
            "$(name)₊c35" => (1.74*d_rs/tau_rs,    ),  # a_rs
            "$(name)₊c36" => (-10.0*d_rs/tau_rs,   ),  # b_rs * x
            "$(name)₊c37" => (-1.0*d_rs/tau_rs,    ),  # β_rs * y
            "$(name)₊c38" => (d_rs/tau_rs,         ),  # input scaling 1
            "$(name)₊c39" => (d_rs/tau_rs,         ),  # input scaling 2
        )

        node.build_setts.new_param_values =
            merge(pop1_params, pop2_params, rs_params)
    end

    return pops, node
end

function EpileptorCore(node::Node;
    a   = 1.0,
    b   = 3.0,
    k   = 1.0,        # slope of right branch
    c_y = 1.0,
    d_y = 5.0,
    r   = 3.5e-4,
    s   = 4.0,
    x0  = -1.6,
    I   = 3.1)

    pops = [
        Population(1, "Core", node;
            input_dynamics_spec  = "slowfast_piecewise_poly_kernel",
            sensory_conn_func    = "linear",
            sends_internode_output = false,
        )
    ]

    node.n_pops = length(pops)
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)

    # single-node LFP: just X
    node.brain_source = "$(node.name)₊x11"

    # ICs near resting state
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        "$(node.name)₊x11" => (-1.8, -1.4),   # X ~ x1
        "$(node.name)₊x12" => (-15.0, -10.0), # Y ~ y1
        "$(node.name)₊x13" => (3.6, 4.0),     # W ~ z
    )

    let name = node.name
        node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
            # --- f_fast(X,W) piecewise -------------------------------------
            # θ_fast = 0
            "$(name)₊c11"  => (0.0, ),
            # left branch: a0 + a1 X + a2 X^2 + a3 X^3 = -b X^2 + a X^3
            "$(name)₊c12"  => (0.0, ),     # a0
            "$(name)₊c13"  => (0.0, ),     # a1
            "$(name)₊c14"  => (-b,  ),     # a2
            "$(name)₊c15"  => (a,   ),     # a3
            # right branch: (b0 + b1 W + b2 W^2) X = -k X
            "$(name)₊c16"  => (-k,  ),     # b0
            "$(name)₊c17"  => (0.0, ),     # b1
            "$(name)₊c18"  => (0.0, ),     # b2

            # --- f_slow ≡ 0 -----------------------------------------------
            "$(name)₊c19"  => (0.0, ),     # θ_slow (ignored)
            "$(name)₊c110" => (0.0, ),
            "$(name)₊c111" => (0.0, ),
            "$(name)₊c112" => (0.0, ),
            "$(name)₊c113" => (0.0, ),
            "$(name)₊c114" => (0.0, ),
            "$(name)₊c115" => (0.0, ),
            "$(name)₊c116" => (0.0, ),
            "$(name)₊c117" => (0.0, ),

            # --- dX = Y - W + I - f_fast(X,W) ------------------------------
            "$(name)₊c118" => (1.0, ),     # A1 * Y
            "$(name)₊c119" => (-1.0,),     # A2 * W
            "$(name)₊c120" => (0.0, ),     # A3 * U (unused)
            "$(name)₊c121" => (I,   ),     # A4
            "$(name)₊c122" => (0.0, ),     # A5 * X
            "$(name)₊c123" => (0.0, ),     # A6 * X^3

            # --- dY = c_y - d_y X^2 - Y -----------------------------------
            "$(name)₊c124" => (-1.0,),     # B1 * Y
            "$(name)₊c125" => (c_y, ),     # B2
            "$(name)₊c126" => (0.0, ),     # B3 * X
            "$(name)₊c127" => (-d_y,),     # B4 * X^2

            # --- dW = r ( s (X - x0) - W ) ---------------------------------
            "$(name)₊c128" => (r,   ),     # C1
            "$(name)₊c129" => (s,   ),     # C2
            "$(name)₊c130" => (x0,  ),     # C3
            "$(name)₊c131" => (0.0, ),     # C4 * U (unused)
        )
    end

    return pops, node
end

function EpileptorOrig(node::Node)

    # ------------------------------------------------------------------
    # Populations:
    #   P1: Epileptor pop1 (x1, y1, z, g)
    #   P2: Epileptor pop2 (x2, y2)
    # ------------------------------------------------------------------
    pops = [
        Population(1, "P1", node;
            input_dynamics_spec   = "epileptor_pop1_dynamics",
            sensory_conn_func     = "linear",
            sends_internode_output = false,
        ),

        Population(2, "P2", node;
            input_dynamics_spec   = "epileptor_pop2_dynamics",
            internode_conn_func   = "linear",
            noise_dynamics_spec   = "additive",
            noise_dynamics        = "rand(Normal(0, 0.00025))",
        ),
    ]

    node.n_pops = length(pops)

    # ------------------------------------------------------------------
    # Population-to-population connectivity
    # ------------------------------------------------------------------
    # For now, no explicit pop-to-pop coupling at the ENMEEG level.
    # The "inputs" variable in each pop will carry sensory / externode
    # inputs only. You can later set one of these entries to "linear"
    # if you want P2 → P1 or P1 → P2 coupling.
    node.build_setts.pop_conn = fill("none", node.n_pops * node.n_pops)
    # Example if you later decide P2 should drive P1 linearly:
    # node.build_setts.pop_conn[2] = "linear"  # depending on your index convention

    # ------------------------------------------------------------------
    # LFP / brain source
    # ------------------------------------------------------------------
    # Simple core LFP: mixture of x2 - x1 (classic Epileptor-ish)
    node.brain_source =
        "0.5 * ($(node.name)₊x21 - $(node.name)₊x11)"

    # ------------------------------------------------------------------
    # Initial conditions
    # Names follow your builders:
    #   P1: x11,x12,x13,x14 = x1,y1,z,g
    #   P2: x21,x22        = x2,y2
    # ------------------------------------------------------------------
    node.build_setts.new_state_var_inits = Dict{String, Tuple{Float64, Float64}}(
        # Pop 1: (x1, y1, z, g)
        "$(node.name)₊x11" => (-1.8,  -1.4),   # x1
        "$(node.name)₊x12" => (-15.0, -10.0),  # y1
        "$(node.name)₊x13" => (3.6,   4.0),    # z
        "$(node.name)₊x14" => (0.0,   0.1),    # g

        # Pop 2: (x2, y2)
        "$(node.name)₊x21" => (-1.1,  -0.9),   # x2
        "$(node.name)₊x22" => (0.001, 0.01),   # y2
    )

    # ------------------------------------------------------------------
    # Parameters
    # These indices match how your builders create Params:
    #
    # epileptor_pop1_dynamics:
    #   c1: a
    #   c2: b
    #   c3: s (slope in f1 right branch)
    #   c4: c  (y1 eq)
    #   c5: d  (y1 eq)
    #   c6: r
    #   c7: s  (slow eq)
    #   c8: x0
    #   c9: I1
    #
    # epileptor_pop2_dynamics:
    #   c1: aa
    #   c2: tau
    #   c3: I2
    #
    # With your naming convention this becomes:
    #   P1: N₊c11 ... N₊c19
    #   P2: N₊c21 ... N₊c23
    # ------------------------------------------------------------------
    let name = node.name
        node.build_setts.new_param_values = Dict{String, Tuple{Vararg{Float64}}}(
            # --- Pop 1 (x1,y1,z,g): epileptor_pop1_dynamics ----------------
            "$(name)₊c11" => (1.0,     ),  # a
            "$(name)₊c12" => (3.0,     ),  # b
            "$(name)₊c13" => (4.0,     ),  # s (in f1 right branch)
            "$(name)₊c14" => (1.0,     ),  # c
            "$(name)₊c15" => (5.0,     ),  # d
            "$(name)₊c16" => (3.5e-4,  ),  # r
            "$(name)₊c17" => (4.0,     ),  # s (in z equation)
            "$(name)₊c18" => (-1.6,    ),  # x0
            "$(name)₊c19" => (3.1,     ),  # I1

            # --- Pop 2 (x2,y2): epileptor_pop2_dynamics --------------------
            "$(name)₊c21" => (6.0,     ),  # aa
            "$(name)₊c22" => (10.0,    ),  # tau
            "$(name)₊c23" => (0.45,    ),  # I2
        )
    end

    return pops, node
end
