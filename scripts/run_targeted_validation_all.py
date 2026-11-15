"""
Automated, parallelized, resumable Step 5 validation over (by default) every
rule in 05_created_rules.json, using the targeted per-rule cluster sampling
added to rule_validation.py (build_targeted_validation_batch) instead of
build_validation_batch's blind random draw from the whole fail_final pool --
confirmed necessary 2026-10-01: a 12-item random batch gave 0 helped/0 harmed
for two strong candidate rules (support 74 and 84) purely from batch-
composition luck; the same two rules got helped=2/6 with 0 harmed once their
own cluster supplied the 'incorrect' half.

Mirrors error_mining.py's concurrency + incremental-checkpoint pattern:
ThreadPoolExecutor across independent units of work (here, rules instead of
questions), one shared correction_llm/generation_llm instance reused by every
worker thread, and every rule's outcome appended to a JSONL checkpoint file
immediately so a killed/restarted job only re-does what's left -- rule_name
is the resume key, exactly like error_mining.py's question_id.
"""
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from dextersql.rule_creator.config import load_rule_creator_config
from dextersql.rule_creator.rule_validation import (
    build_targeted_validation_batch,
    _find_group_for_rule,
    _validate_once,
    _passes,
)
from dextersql.rule_creator.prompts import format_rule_refinement_prompt
from dextersql.rule_creator.parsing import parse_rule_refinement
from dextersql.core.llm import make_llm
from dextersql.core.services import configure_schema_service
from dextersql.core.logger import logger

N_INCORRECT = 30
N_CORRECT = 16
N_PARALLEL = 12
CHECKPOINT_NAME = "06_validation_checkpoint.jsonl"
FINAL_ACCEPTED_NAME = "06_validated_rules_final.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/rule_creator.toml", help="rule creator config (see config/rule_creator.toml.template)")
parser.add_argument("--only", nargs="*", default=None, help="validate only these rule_names (default: all)")
parser.add_argument("--seed-accepted", nargs="*", default=[], help="rule_names already known-accepted from a prior probe -- written straight to the checkpoint without re-validating")
args = parser.parse_args()

config = load_rule_creator_config(args.config)
out_dir = Path(config.output_dir)
checkpoint_path = out_dir / CHECKPOINT_NAME

all_rules = json.load(open(config.created_rules_path))
rules_by_name = {r["rule_name"]: r for r in all_rules}

target_names = args.only if args.only else list(rules_by_name.keys())
target_names = [n for n in target_names if n in rules_by_name]

already_done = set()
if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                already_done.add(json.loads(line)["rule_name"])

todo = [n for n in target_names if n not in already_done]
print(f"[validate_all] {len(target_names)} targeted, {len(already_done)} already done, {len(todo)} to process", flush=True)
print(f"[validate_all] n_incorrect={N_INCORRECT} n_correct={N_CORRECT} n_parallel={N_PARALLEL} "
      f"harm_ratio_threshold={config.rule_validation.harm_ratio_threshold} max_refine_rounds={config.rule_validation.max_refine_rounds}", flush=True)

configure_schema_service()
correction_llm = make_llm(config.rule_validation.llm)
generation_llm = make_llm(config.error_mining.generation_llm)

checkpoint_f = open(checkpoint_path, "a")


def _write_result(result):
    checkpoint_f.write(json.dumps(result) + "\n")
    checkpoint_f.flush()


# pre-seed already-known-accepted rules (from the earlier 2-rule probe) without re-validating
for name in args.seed_accepted:
    if name in already_done:
        continue
    rule = rules_by_name.get(name)
    if rule is None:
        print(f"[validate_all] WARNING: --seed-accepted rule {name!r} not found, skipping", flush=True)
        continue
    _write_result({
        "rule_name": name, "accepted": True, "final_rule": rule,
        "source": "seeded_from_prior_probe", "rounds": [],
    })
    already_done.add(name)
    todo = [n for n in todo if n != name]
    print(f"[validate_all] seeded {name} as already-accepted (prior probe)", flush=True)


def validate_one_rule(rule):
    name = rule["rule_name"]
    try:
        group = _find_group_for_rule(rule, config.error_groups_path)
    except Exception as e:
        return {"rule_name": name, "accepted": False, "final_rule": None, "error": f"group lookup failed: {e}", "rounds": []}

    try:
        batch = build_targeted_validation_batch(config, rule, generation_llm, N_INCORRECT, N_CORRECT, seed=0)
    except Exception as e:
        return {"rule_name": name, "accepted": False, "final_rule": None, "error": f"batch build failed: {e}",
                "group_id": group["group_id"], "group_label": group["label"], "rounds": []}

    current_rule = dict(rule)
    round_logs = []
    accepted = False
    for round_idx in range(config.rule_validation.max_refine_rounds + 1):
        cases = _validate_once(current_rule, batch, correction_llm, config)
        n_helped, n_harmed = len(cases["helped"]), len(cases["harmed"])
        harm_ratio = (n_harmed / n_helped) if n_helped else None
        round_logs.append({
            "round": round_idx, "helped": n_helped, "harmed": n_harmed,
            "unchanged": len(cases["unchanged"]), "harm_ratio": harm_ratio,
            "helped_examples": cases["helped"][:3], "harmed_examples": cases["harmed"][:3],
        })
        print(f"[validate_all]   {name} round={round_idx} helped={n_helped} harmed={n_harmed} "
              f"harm_ratio={'inf' if harm_ratio is None else f'{harm_ratio:.2f}'}", flush=True)

        if _passes(cases, config.rule_validation.harm_ratio_threshold):
            accepted = True
            print(f"[validate_all]   {name} ACCEPTED (round {round_idx})", flush=True)
            break
        if round_idx == config.rule_validation.max_refine_rounds:
            print(f"[validate_all]   {name} DISCARDED after {round_idx} refine rounds", flush=True)
            break

        refine_prompt = format_rule_refinement_prompt(current_rule, cases["helped"], cases["harmed"], cases["unchanged"])
        try:
            responses, _usage = correction_llm.ask([{"role": "user", "content": refine_prompt}], n=1, temperature=0.7, timeout=90)
            refined = parse_rule_refinement(responses[0].content)
        except Exception as e:
            logger.warning(f"[validate_all] refinement call failed for {name}: {e}")
            refined = None
        if refined is None:
            print(f"[validate_all]   {name} refinement call failed to parse, stopping early -> discarded", flush=True)
            break
        refined["rule_name"] = name
        current_rule = refined

    return {
        "rule_name": name, "accepted": accepted, "final_rule": current_rule if accepted else None,
        "group_id": group["group_id"], "group_label": group["label"],
        "n_cluster_members": len(group["member_rows"]), "batch_size": len(batch),
        "rounds": round_logs,
    }


completed = 0
with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
    futures = {pool.submit(validate_one_rule, rules_by_name[name]): name for name in todo}
    for future in as_completed(futures):
        name = futures[future]
        try:
            result = future.result()
        except Exception as e:
            logger.warning(f"[validate_all] {name} raised unexpectedly: {e}")
            result = {"rule_name": name, "accepted": False, "final_rule": None, "error": str(e), "rounds": []}
        _write_result(result)
        completed += 1
        if completed % 5 == 0 or completed == len(todo):
            print(f"[validate_all] {completed}/{len(todo)} rules processed", flush=True)

checkpoint_f.close()

# consolidate final accepted set from the full checkpoint (includes seeded + this run)
all_results = []
with open(checkpoint_path) as f:
    for line in f:
        line = line.strip()
        if line:
            all_results.append(json.loads(line))

accepted_rules = [r["final_rule"] for r in all_results if r.get("accepted") and r.get("final_rule")]
final_path = out_dir / FINAL_ACCEPTED_NAME
with open(final_path, "w") as f:
    json.dump(accepted_rules, f, indent=2)

print(f"[validate_all] DONE: {len(accepted_rules)}/{len(all_results)} rules accepted overall -> {final_path}", flush=True)
