# Example 3: Grammar-Based Model Sampling
# ============================================================================
# Demonstrates how to sample neural network model structures from a probabilistic context-free grammar (PCFG).
# This uses the grammar defined in grammars/default_grammar.cfg to generate diverse model architectures.
using ENEEGMA
using Plots

println("\nENEEGMA Grammar-Based Model Sampling Example\n")

# ============================================================================
# Step 1: Create Settings with Defaults
# ============================================================================
settings = create_default_settings();
print_settings_summary(settings; section="sampling_settings")
settings.general_settings.seed = 1  # Set seed for reproducible grammar sampling

# ============================================================================
# Step 2: Sample Models from Grammar
# ============================================================================
# Sample models from grammar using settings configuration
# sample_from_grammar(settings) automatically extracts:
#   - grammar_file: path to grammar (from settings.sampling_settings.grammar_file)
#   - n_samples: number of models to generate (from settings.sampling_settings.n_samples)
#   - grammar_seed: RNG seed for reproducibility, with seed priority:
#       1. settings.sampling_settings.grammar_seed (if not nothing)
#       2. settings.general_settings.seed (fallback)
#       3. nothing (random seed)
#
# Returns a Dict with three model representations:
#   :full_rule_expansion => Vector of OrderedDict{String,String} - model structure as (position:symbol, rhs) pairs
#   :parse_tree          => Vector of RuleTree - hierarchical parse tree representation (Each node stores rule_id and child subtrees, serializable to compact string format)
#   :terminals           => Vector of String - flattened terminal sequence representation
candidate_models = sample_from_grammar(settings)

# ============================================================================
# CHECK MODEL STRUCTURE
# Extract the first sampled model in different representations:
# ============================================================================
sampled_rule_expansion = candidate_models[:full_rule_expansion][1]
sampled_parsed_rule_tree = candidate_models[:parse_tree][1]
sampled_terminals = candidate_models[:terminals][1]

# ============================================================================
# SAVE the sampled model parse trees to CSV file 
# The models dynamics / equations are later build from the parse trees.
# ============================================================================
save_parse_trees(candidate_models[:parse_tree], settings)

# ============================================================================
# BUILD SAMPLED MODEL
# ============================================================================
println("\n--- Checking Candidate Model 1 ---")
i = 1
settings.network_settings.name = "G$i"
settings.network_settings.node_models = [candidate_models[:parse_tree][i]]

net = build_network(settings)

# Display network summary
display(net)
