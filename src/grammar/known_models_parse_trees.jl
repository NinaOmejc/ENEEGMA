#= 
####################################################################
###########                 Grammar gs                      ########
####################################################################

terminals_gs = Dict(
    "WC" => ["exp_kernel", "direct_readout", "custom", "custom", 
             "exp_kernel", "direct_readout", "custom", "false", 
             "baseline_sigmoid", "custom", "custom", "full"],
    "ARM" => ["second_order_kernel", "direct_readout", "linear", "linear",  
              "second_order_kernel", "direct_readout", "linear", "false", 
              "baseline_sigmoid", "linear", "linear", "ring"],
    "JR" => ["second_order_kernel", "direct_readout", "false", "false",
             "second_order_kernel", "direct_readout", "linear", "linear",  
             "second_order_kernel", "direct_readout", "false", "false",  
             "saturating_sigmoid", "custom", "custom", "custom", "star"],
    "W" => ["second_order_kernel", "direct_readout", "false", "false",
            "second_order_kernel", "direct_readout", "linear", "linear",  
            "second_order_kernel", "direct_readout", "false", "false",  
            "second_order_kernel", "direct_readout", "false", "false",  
            "second_order_kernel", "direct_readout", "false", "false",  
            "saturating_sigmoid", "custom", "custom", "custom", "custom", "custom", "star_feedback_tail"],
    "MDF" => ["second_order_kernel", "second_order_kernel", "direct_readout", "false", "false", "false", "false",
              "second_order_kernel", "second_order_kernel", "direct_readout", "false", "false", "false", "false",
              "second_order_kernel", "direct_readout", "linear", "linear",
              "baseline_sigmoid", "custom", "custom", "custom", "custom", "custom", "multi_cycle"],
    "LW" => ["second_order_kernel", "second_order_kernel", "membrane_integrator", "linear", "false", "linear", "false",
             "second_order_kernel", "second_order_kernel", "membrane_integrator", "false", "linear", "false", "false",
             "saturating_sigmoid", "custom", "custom", "custom", "custom", "chain_loop"],
    "RRWC" => ["second_order_kernel", "spatial_gradient", "false", "linear", 
               "second_order_kernel", "direct_readout", "false", "linear", 
               "saturating_sigmoid", "linear", "custom", "full"],
    "RRWT" => ["second_order_kernel", "direct_readout", "linear", "linear", 
               "second_order_kernel", "direct_readout", "false", "linear", 
               "saturating_sigmoid", "custom", "custom", "ring"],
    "WW" => ["gating_kinetics", "direct_readout", "custom", "custom", 
             "exp_kernel", "direct_readout", "custom", "custom", 
             "relaxed_rectifier", "custom", "custom", "full"],
    "WWR" => ["gating_kinetics", "direct_readout", "custom", "custom", 
              "relaxed_rectifier", "custom", "full"],
    "LB" => ["voltage_gated_dynamics", "direct_readout", "linear", "linear",
             "saturating_sigmoid", "linear", "null"],   
    "HO" => ["second_order_kernel", "direct_readout", "linear", "linear",
             "saturating_sigmoid", "linear", "null"],
    "MPR" => ["x1", "*", "x2", 
              "x1", "*", "x1", "+", "x2", "*", "x2", 
              "direct_readout", "linear", "linear", "saturating_sigmoid", "linear", "null"],
    "FHN" => ["x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", 
              "x1", "+", "x2",
              "direct_readout", "linear", "linear", "saturating_sigmoid", "linear", "null"],
    "VDP" => ["x2", 
              "x1", "+", "x2", "+", "x1", "*", "x1" , "*", "x2",
              "direct_readout", "linear", "linear", "saturating_sigmoid", "linear", "null"],
    "SL" => ["x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", "+", "x1", "*", "x2", "*", "x2",
             "x1", "+", "x2", "+", "x2", "*", "x2", "*", "x2", "+", "x1", "*", "x1", "*", "x2",
             "direct_readout", "linear", "linear", "saturating_sigmoid", "linear", "null"],
    "DO" => ["x2", "x1", "+", "x2", "+", "x1", "*", "x1", "*", "x1", 
             "direct_readout", "linear", "linear", "saturating_sigmoid", "linear", "null"],
    "PD" => ["linear_kernel", "direct_readout", "linear", "custom",
             "fourier_basis", "linear", "null"]
    
)


parse_trees_gs = Dict(
  "WC"   => "2{6{8{12,16,21,21},8{12,16,21,22}},27,23,23,37}",
  "ARM"  => "2{6{8{11,16,20,20},8{11,16,20,22}},27,24,24,38}",
  "JR"   => "3{8{11,16,22,22},6{8{11,16,20,20},8{11,16,22,22}},25,23,23,23,42}",
  "W"    => "5{6{8{11,16,22,22},8{11,16,20,20}},6{8{11,16,22,22},8{11,16,22,22}},8{11,16,22,22},25,23,23,23,23,23,42}",
  "MDF"  => "5{7{9{11,11,16,22,22,22,22}},7{9{11,11,16,22,22,22,22}},8{11,16,20,20},27,23,23,23,23,23,45}",
  "LW"   => "4{7{9{11,11,17,20,22,20,22}},7{9{11,11,17,22,20,22,22}},25,23,23,23,23,46}",
  "RRWC" => "2{6{8{11,19,22,20},8{11,16,22,20}},25,24,23,37}",
  "RRWT" => "2{6{8{11,16,20,20},8{11,16,22,20}},25,23,23,38}",
  "WW"   => "2{6{8{13,16,21,21},8{12,16,21,21}},26,23,23,37}",
  "WWR"  => "1{8{13,16,21,21},26,23,35}",
  "LB"   => "1{8{15,16,20,20},25,24,36}",
  "HO"   => "1{8{11,16,20,20},25,24,36}",
  "MPR"  => "1{8{10{30{31{33,32{34}}},29{31{33,32{33}},30{31{34,32{34}}}}},16,20,20},25,24,36}",
  "FHN"  => "1{8{10{29{32{33},29{32{34},30{31{33,31{33,32{33}}}}}},29{32{33},30{32{34}}}},16,20,20},25,24,36}",     
  "VDP"  => "1{8{10{30{32{34}},29{32{33},29{32{34},30{31{33,31{33,32{34}}}}}}},16,20,20},25,24,36}",
  "SL"   => "1{8{10{29{32{33},29{32{34},29{31{33,31{33,32{33}}},30{31{33,31{34,32{34}}}}}}},29{32{33},29{32{34},29{31{34,31{34,32{34}}},30{31{33,31{33,32{34}}}}}}}},16,20,20},25,24,36}",
  "DO"   => "1{8{10{30{32{34}},29{32{33},29{32{34},30{31{33,31{33,32{33}}}}}}},16,20,20},26,24,36}",
  "PD"   => "1{8{14,16,20,21},28,24,36}",
)
 =#
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

# with grammar and full CM structure
parse_trees_gb = Dict(
    "WC" => "",
    "ARM" => "",
    "JR" => "",
    "W" => "",
    "MDF" => "",
    "LW" => "",
    "RRWC" => "",
    "RRWT" => "",
    "WW" => "",
    "WWR" => "",
    "LB" => "",
    "HO" => "",
    "MPR" => "",
    "FHN" => "",
    "VDP" => "",
    "SL" => "",
    "DO" => "",
    "PD" => "",
)

# without grammar, only polynomial, and with short, modular PC structure
parse_trees_ps = Dict(
    "WC" => "",
    "ARM" => "",
    "JR" => "",
    "W" => "",
    "MDF" => "",
    "LW" => "",
    "RRWC" => "",
    "RRWT" => "",
    "WW" => "",
    "WWR" => "",
    "LB" => "",
    "HO" => "",
    "MPR" => "",
    "FHN" => "",
    "VDP" => "",
    "SL" => "",
    "DO" => "",
    "PD" => "",
)

# without grammar, only polynomial, and with full PC structure
parse_trees_pb = Dict(
    "WC" => "",
    "ARM" => "",
    "JR" => "",
    "W" => "",
    "MDF" => "",
    "LW" => "",
    "RRWC" => "",
    "RRWT" => "",
    "WW" => "",
    "WWR" => "",
    "LB" => "",
    "HO" => "",
    "MPR" => "",
    "FHN" => "",
    "VDP" => "",
    "SL" => "",
    "DO" => "",
    "PD" => "",
)

