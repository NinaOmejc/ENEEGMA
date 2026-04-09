####################################################################
###########                 Grammar gsn                     ########
####################################################################

terminals_gsn = Dict(
    "WC" => ["exp_kernel", "direct_readout", "custom", "custom", "false",
             "exp_kernel", "direct_readout", "custom", "false", "false",
             "baseline_sigmoid", "custom", "custom", "full"],

    "ARM" => ["second_order_kernel", "direct_readout", "linear", "linear", "true", 
              "second_order_kernel", "direct_readout", "linear", "false", "true", 
              "any", "linear", "linear", "ring"],   # CF is "any": ARM's pop_conn is linear, not representable in grammar_gsn CF rule

    "JR" => ["second_order_kernel", "direct_readout", "false", "false", "false", 
             "second_order_kernel", "direct_readout", "linear", "linear", "false",   
             "second_order_kernel", "direct_readout", "false", "false", "false", 
             "saturating_sigmoid", "custom", "custom", "custom", "star"],

    "W" => ["second_order_kernel", "direct_readout", "false", "false", "false", 
            "second_order_kernel", "direct_readout", "linear", "linear", "false",  
            "second_order_kernel", "direct_readout", "false", "false", "false",  
            "second_order_kernel", "direct_readout", "false", "false", "false", 
            "second_order_kernel", "direct_readout", "false", "false", "false",  
            "saturating_sigmoid", "custom", "custom", "custom", "custom", "custom", "star_feedback_tail"],

    "MDF" => ["second_order_kernel", "second_order_kernel", "difference", "false", "false", "false", "false", "false", 
              "second_order_kernel", "second_order_kernel", "difference", "false", "false", "false", "false", "false", 
              "second_order_kernel", "direct_readout", "linear", "linear", "false", 
              "baseline_sigmoid", "custom", "custom", "custom", "custom", "custom", "star_loop_extended"],

    "LW" => ["second_order_kernel", "second_order_kernel", "membrane_integrator", "linear", "false", "linear", "false", "true",   # Pop1 E: noise=stochastic -> true
             "second_order_kernel", "second_order_kernel", "membrane_integrator", "false", "linear", "false", "false", "false",  # Pop2 I: no noise
             "saturating_sigmoid", "custom", "custom", "custom", "custom", "ei_extended"],

    "RRWC" => ["second_order_kernel", "spatial_gradient", "false", "linear", "false",  
               "second_order_kernel", "direct_readout", "false", "linear", "false", 
               "saturating_sigmoid", "linear", "custom", "full"],

    "RRWT" => ["second_order_kernel", "direct_readout", "linear", "linear", "false", 
               "second_order_kernel", "direct_readout", "false", "linear", "false", 
               "saturating_sigmoid", "custom", "custom", "ring"],

    "WW" => ["gating_kinetics", "direct_readout", "custom", "custom", "true", 
             "exp_kernel", "direct_readout", "custom", "custom", "true", 
             "relaxed_rectifier", "custom", "custom", "full"],

    "WWR" => ["gating_kinetics", "direct_readout", "custom", "custom", "true", 
              "relaxed_rectifier", "custom", "full"],

    "LB" => ["voltage_gated_dynamics", "direct_readout", "linear", "linear", "false", 
             "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...), not representable in grammar_gsn CF rule

    "HO" => ["second_order_kernel", "direct_readout", "linear", "linear", "false", 
             "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...), not representable in grammar_gsn CF rule

    "MPR" => ["x1", "*", "x2", 
              "x1", "+", "x1", "*", "x1", "+", "x2", "*", "x2", 
              "direct_readout", "linear", "linear", "false", "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...)

    "FHN" => ["x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", 
              "x1", "+", "x2",
              "direct_readout", "linear", "linear", "false", "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...)

    "VDP" => ["x2", 
              "x1", "+", "x2", "+", "x1", "*", "x1" , "*", "x2",
              "direct_readout", "linear", "linear", "false", "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...)

    "SL" => ["x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", "+", "x1", "*", "x2", "*", "x2",
             "x1", "+", "x2", "+", "x2", "*", "x2", "*", "x2", "+", "x1", "*", "x1", "*", "x2",
             "direct_readout", "linear", "linear", "false", "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...)

    "DO" => ["x2", "x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", 
             "direct_readout", "linear", "linear", "false", "any", "linear", "null"],   # CF is "any": pop_conn=fill("none",...)
)
