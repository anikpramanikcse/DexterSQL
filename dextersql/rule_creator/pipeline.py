"""
Rule Creator orchestrator and CLI.

Runs the five steps in order, each reading the previous step's output file
and writing its own -- so the pipeline can be resumed or re-run from any
step without repeating earlier (expensive) LLM work:

  1. sampling.sample_training_questions      -> 01_sampled_questions.json
  2. error_mining.run_error_mining           -> 02_fail_candidate.jsonl
  3. db_agnostic_filter.run_db_agnostic_filter -> 03_fail_final.jsonl
     clustering.run_clustering               -> 04_error_groups.json
  4. rule_synthesis.run_rule_synthesis       -> 05_created_rules.json
  5. rule_validation.run_rule_validation     -> 06_validated_rules.json

(Step numbers 1-4 above match the paper's four-step methodology; file
numbers match pipeline order, where Step 3 alone produces two files -- see
config.py's DbAgnosticFilterConfig/ClusteringConfig split. Step 5 here is
the second half of the paper's own Step 4 -- "synthesis" and "validation
and refinement" are one paper step, split into two resumable pipeline
steps, since validation is itself expensive and execution-heavy and is
worth checkpointing independently of synthesis.)

Usage:
    python -m dextersql.rule_creator.pipeline --config path/to/rule_creator.toml
    python -m dextersql.rule_creator.pipeline --config ... --start-step 3
    python -m dextersql.rule_creator.pipeline --config ... --stop-step 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dextersql.rule_creator.config import RuleCreatorConfig, load_rule_creator_config
from dextersql.rule_creator.db_agnostic_filter import run_db_agnostic_filter
from dextersql.rule_creator.clustering import run_clustering
from dextersql.rule_creator.error_mining import run_error_mining
from dextersql.rule_creator.rule_synthesis import run_rule_synthesis
from dextersql.rule_creator.rule_validation import run_rule_validation, run_targeted_rule_validation, _load_created_rules
from dextersql.rule_creator.sampling import sample_training_questions

STEP_NAMES = {
    1: "error_mining (sampling + generation + explanation)",
    2: "db_agnostic_filter",
    3: "clustering",
    4: "rule_synthesis",
    5: "rule_validation",
}


def run_pipeline(
    config: RuleCreatorConfig,
    start_step: int = 1,
    stop_step: int = 5,
    only_rule_names: Optional[List[str]] = None,
    seed_accepted: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """only_rule_names/seed_accepted only affect Step 5, and only take effect
    when start_step <= 5 <= stop_step. only_rule_names restricts validation to
    a named subset of 05_created_rules.json (e.g. for a quick probe of a
    hand-picked rule or two); seed_accepted writes already-known-accepted
    rules straight to the checkpoint without re-validating (e.g. results
    carried over from an earlier probe run)."""
    print(f"{'=' * 70}\nRule Creator: steps {start_step}-{stop_step}\n{'=' * 70}")
    print(f"train_json_path:       {config.train_json_path}")
    print(f"train_databases_root:  {config.train_databases_root}")
    print(f"output_dir:            {config.output_dir}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    samples = None
    fail_candidates = None
    fail_final = None
    error_groups = None
    rules = None

    if start_step <= 1:
        print(f"\n--- Step 1: {STEP_NAMES[1]} ---")
        samples = sample_training_questions(config)
        fail_candidates = run_error_mining(config, samples)
    if stop_step < 2:
        return

    if start_step <= 2:
        print(f"\n--- Step 2: {STEP_NAMES[2]} ---")
        fail_final = run_db_agnostic_filter(config, fail_candidates)
    if stop_step < 3:
        return

    if start_step <= 3:
        print(f"\n--- Step 3: {STEP_NAMES[3]} ---")
        error_groups = run_clustering(config, fail_final)
    if stop_step < 4:
        return

    if start_step <= 4:
        print(f"\n--- Step 4: {STEP_NAMES[4]} ---")
        rules = run_rule_synthesis(config, error_groups)
    if stop_step < 5:
        return

    if start_step <= 5:
        print(f"\n--- Step 5: {STEP_NAMES[5]} ---")
        step5_rules = rules
        if only_rule_names:
            pool = rules if rules is not None else _load_created_rules(config.created_rules_path)
            pool_by_name = {r["rule_name"]: r for r in pool}
            missing = [n for n in only_rule_names if n not in pool_by_name]
            if missing:
                raise ValueError(f"--only-rules names not found in {config.created_rules_path}: {missing}")
            step5_rules = [pool_by_name[n] for n in only_rule_names]
        if config.rule_validation.sampling_strategy == "random":
            run_rule_validation(config, step5_rules)
        else:
            run_targeted_rule_validation(config, step5_rules, seed_accepted=seed_accepted)

    print(f"\n{'=' * 70}\nDONE. Validated rules -> {config.validated_rules_path}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule Creator: mine database-agnostic SQL-generation error rules from BIRD training data.")
    parser.add_argument("--config", required=True, help="Path to a rule_creator.toml config (see config/rule_creator.toml.template)")
    parser.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Resume from this step, reusing earlier steps' saved output files")
    parser.add_argument("--stop-step", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Stop after this step")
    parser.add_argument("--only-rules", nargs="*", default=None, metavar="RULE_NAME",
                         help="Step 5 only: validate just these rule_names from 05_created_rules.json instead of all of them")
    parser.add_argument("--seed-accepted-from", default=None, metavar="PATH",
                         help="Step 5 only: path to a JSON list of already-accepted rule dicts (e.g. from a prior probe's "
                              "06_validated_rules.json) to write straight to the checkpoint as accepted, without re-validating")
    args = parser.parse_args()

    config = load_rule_creator_config(args.config)
    seed_accepted = None
    if args.seed_accepted_from:
        with open(args.seed_accepted_from) as f:
            seed_accepted = {r["rule_name"]: r for r in json.load(f)}

    run_pipeline(
        config, start_step=args.start_step, stop_step=args.stop_step,
        only_rule_names=args.only_rules, seed_accepted=seed_accepted,
    )


if __name__ == "__main__":
    main()
