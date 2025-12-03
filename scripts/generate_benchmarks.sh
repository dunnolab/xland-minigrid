# This can take a lot of time. Generate only (when) needed!

# trivial
python scripts/ruleset_generator.py \
  --chain_depth=0 \
  --num_distractor_objects=3 \
  --total_rulesets=21_000 \
  --save_path="trivial_21k"

# small
python scripts/ruleset_generator.py \
  --prune_chain \
  --prune_prob=0.3 \
  --chain_depth=1 \
  --sample_distractor_rules \
  --num_distractor_rules=2 \
  --num_distractor_objects=2 \
  --total_rulesets=1_000_000 \
  --save_path="small_1m"

# medium
python scripts/ruleset_generator.py \
  --prune_chain \
  --prune_prob=0.1 \
  --chain_depth=2 \
  --sample_distractor_rules \
  --num_distractor_rules=3 \
  --num_distractor_objects=2 \
  --total_rulesets=1_000_000 \
  --save_path="medium_1m"

# high
python scripts/ruleset_generator.py \
  --prune_chain \
  --prune_prob=0.1 \
  --chain_depth=3 \
  --sample_distractor_rules \
  --num_distractor_rules=4 \
  --num_distractor_objects=1 \
  --total_rulesets=1_000_000 \
  --save_path="high_1m"

## medium + distractors
#python scripts/ruleset_generator.py \
#  --prune_chain \
#  --prune_prob=0.8 \
#  --chain_depth=2 \
#  --sample_distractor_rules \
#  --num_distractor_rules=4 \
#  --num_distractor_objects=2 \
#  --total_rulesets=1_000_000 \
#  --save_path="medium_dist_1m"

# medium 3M
python scripts/ruleset_generator.py \
  --prune_chain \
  --prune_prob=0.1 \
  --chain_depth=2 \
  --sample_distractor_rules \
  --num_distractor_rules=3 \
  --num_distractor_objects=2 \
  --total_rulesets=3_000_000 \
  --save_path="medium_3m"

# high 3M
python scripts/ruleset_generator.py \
  --prune_chain \
  --prune_prob=0.1 \
  --chain_depth=3 \
  --sample_distractor_rules \
  --num_distractor_rules=4 \
  --num_distractor_objects=1 \
  --total_rulesets=3_000_000 \
  --save_path="high_3m"

