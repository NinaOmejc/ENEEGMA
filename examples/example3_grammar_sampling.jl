# Example 3: Grammar-Based Model Sampling
# ==========================================
# Demonstrates how to sample neural network model structures from a probabilistic context-free grammar (PCFG).
# This uses the grammar defined in grammars/default_grammar.cfg to generate diverse model architectures.
using ENEEGMA
using Plots

# Create settings with defaults
# By default: n_samples=10, grammar_file="grammars/default_grammar.cfg", grammar_seed=nothing
settings = create_default_settings();
# Set seed for reproducible grammar sampling
# Option 1: Set general_settings.seed (used as fallback if grammar_seed is nothing)
settings.general_settings.seed = 1
# Option 2: Set sampling_settings.grammar_seed directly (takes priority over general_settings.seed)
# settings.sampling_settings.grammar_seed = 1

# Print sampling configuration to see current settings
# Shows: grammar_file path, n_samples (number of models to generate), only_unique flag, max_resample_attempts
print_settings_summary(settings; section="sampling_settings")

# Sample models from grammar using settings configuration
# sample_from_grammar(settings) automatically extracts:
#   - grammar_file: path to grammar (from settings.sampling_settings.grammar_file)
#   - n_samples: number of models to generate (from settings.sampling_settings.n_samples)
#   - grammar_seed: RNG seed for reproducibility, with seed priority:
#       1. settings.sampling_settings.grammar_seed (if not nothing)
#       2. settings.general_settings.seed (fallback)
#       3. nothing (random seed)
#
# Returns a Dict with three representations:
#   :full_rule_expansion => Vector of OrderedDict{String,String} - model structure as (position:symbol, rhs) pairs
#   :parse_tree          => Vector of RuleTree - hierarchical parse tree representation
#   :terminals           => Vector of String - flattened terminal sequence representation
candidate_models = sample_from_grammar(settings)

# Extract the first sampled model in different representations:

# sampled_rule_expansion: OrderedDict with rule expansion history
# Keys like "0:Node", "1:InputDyn" represent (position:LHS_symbol)
# Values are the RHS (rule expansion)
sampled_rule_expansion = candidate_models[:full_rule_expansion][1]

# sampled_parsed_rule_tree: RuleTree containing the full hierarchical parse tree
# Each node stores rule_id and child subtrees, serializable to compact string format
# Can be converted to/from string format: "2{6{12,16,21},6{12,16,21},25,23}"
sampled_parsed_rule_tree = candidate_models[:parse_tree][1]

# sampled_terminals: Space-separated terminal sequence produced by the grammar
# This is the final "flattened" model specification
sampled_terminals = candidate_models[:terminals][1]

# save the sampled model parse trees to CSV file
# Output: model_name (G1, G2, ...) and parse_tree (serialized parse trees)
save_parse_trees(candidate_models[:parse_tree], settings)

# Build and test each candidate model separately using candidate_name
# Each candidate will be saved to its own folder: path_out/exp_name/candidate_G1/, candidate_G2/, etc.
# Files will also include the candidate name: SimpleNetwork_G1_equations.tex, SimpleNetwork_G2_params_inits_run1.csv, etc.
println("\n--- Testing Candidate Models ---")
i = 1
model_id = "G$i"
settings.network_settings.node_models = [candidate_models[:parse_tree][i]]
settings.general_settings.candidate_name = model_id

println("\nBuilding network for candidate $model_id...")
net = build_network(settings)
