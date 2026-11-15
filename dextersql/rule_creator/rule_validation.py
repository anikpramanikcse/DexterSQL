"""
Step 5 (Rule validation and refinement) -- second half of the paper's
Section 3.1.4 Step 4.

A synthesized rule may contain incorrect error reasoning or patterns,
hallucinated instructions, or otherwise be suboptimal -- applying it could
harm a generated SQL instead of correcting it. This step samples ONE mixed
batch of correct and incorrect candidate SQLs from D_train (reused across
every rule and every refinement round, so accept/reject decisions stay
comparable), reuses the SAME rule-relevance-selection + correction call
already implemented in dextersql/rule_correction/ (invoked standalone --
one rule at a time, directly via format_rule_correction_prompt -- rather
than through SQLRuleCorrectionRunner's full pipeline machinery, which
expects a live DataItem/dataset this offline step doesn't have) to find
which batch items each rule applies to and to get each one's corrected SQL,
executes every corrected SQL, and compares it to gold.

A case is helped when an originally-incorrect candidate becomes correct
after applying the rule; harmed when an originally-correct candidate
becomes incorrect. A rule is accepted iff it helps at least one candidate
and harm_ratio = |harmed| / |helped| is under
config.rule_validation.harm_ratio_threshold. A rejected rule is sent back
to an LLM for refinement with its helped/harmed/unchanged cases as
context, then re-validated against the SAME batch, for up to
config.rule_validation.max_refine_rounds rounds; a rule that never clears
the threshold is discarded.

The correct half of the batch cannot be reused from any existing Step 1
artifact -- error_mining.py explicitly discards every candidate that
matches gold ("correct, discard per the methodology"). This step reruns
that same generation call on a fresh sample of training questions and
keeps the opposite outcome.

Input: config.created_rules_path (from rule_synthesis.run_rule_synthesis).
Output: config.validated_rules_path -- only the rules that passed.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from dextersql.core.db_utils import execute_sql as _execute_sql_unbounded
from dextersql.core.llm import make_llm
from dextersql.core.logger import logger
from dextersql.core.services import configure_schema_service, get_schema_service
from dextersql.rule_correction.prompts import format_rule_correction_prompt
from dextersql.rule_creator.config import RuleCreatorConfig
from dextersql.rule_creator.parsing import (
    parse_rule_correction_shape,
    parse_rule_refinement,
)
from dextersql.rule_creator.prompts import format_rule_refinement_prompt, format_sql_generation_prompt
from dextersql.rule_creator.parsing import parse_sql_result

_MAX_FORMAT_RETRY = 3  # malformed-<result> retries; NOT retried for a well-formed "rule doesn't apply"

# execute_sql's own timeout is a time-based progress-handler check during
# SQLite VM steps -- it does not reliably bound a fetchall() over an
# already-huge result set (the same root cause behind the two OOM crashes on
# the BIRD repair run diagnosed earlier). A rule-corrected SQL here is
# untrusted (an LLM guess), so a wrong "fix" can turn a cheap query into an
# unbounded scan/cross-join. Rather than touch the shared execute_sql
# implementation, every call in this module goes through a hard wall-clock
# guard: the future is abandoned (not killed -- CPython can't forcibly kill a
# blocked C-level sqlite3 call) if it overruns, so this module's own control
# flow never blocks past _SQL_HARD_TIMEOUT regardless of what the query does.
# Pool size > 1 is load-bearing: an abandoned call leaves its worker thread
# permanently blocked (leaked, never returns), so a single-worker pool would
# let one bad query silently wedge every later call behind it forever.
_SQL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rule_validation_sql")
_SQL_HARD_TIMEOUT = 20


def execute_sql(db_path: str, sql: str, timeout: int = 30):
    future = _SQL_EXECUTOR.submit(_execute_sql_unbounded, db_path, sql, timeout=timeout)
    try:
        return future.result(timeout=_SQL_HARD_TIMEOUT)
    except FutureTimeoutError:
        logger.warning(f"[rule_validation] execute_sql exceeded {_SQL_HARD_TIMEOUT}s hard timeout, treating as failed: {sql[:200]}")
        return SimpleNamespace(result_rows=None)
    except Exception as e:
        logger.warning(f"[rule_validation] execute_sql raised: {e}")
        return SimpleNamespace(result_rows=None)


def _load_created_rules(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _load_fail_final(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_train_rows(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _db_path(config: RuleCreatorConfig, db_id: str) -> str:
    return str(Path(config.train_databases_root) / db_id / f"{db_id}.sqlite")


def _normalize_sql(sql: str) -> str:
    return " ".join((sql or "").split()).strip().lower()


def _rows_equal(a, b) -> bool:
    return a.result_rows is not None and b.result_rows is not None and set(a.result_rows) == set(b.result_rows)


def _build_correct_candidate_half(
    config: RuleCreatorConfig, generation_llm, n_target: int, seed: int
) -> List[Dict[str, Any]]:
    """Reruns Step 1's own generation call on a fresh sample of training
    questions, keeping only the candidate that MATCHES gold -- the exact
    opposite of what error_mining.py keeps, and the only way to obtain a
    correct candidate at all, since Step 1 never saves one."""
    rows = _load_train_rows(config.train_json_path)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    correct_half: List[Dict[str, Any]] = []
    for idx in order:
        if len(correct_half) >= n_target:
            break
        row = rows[idx]
        db_id = row.get("db_id")
        gold_sql = row.get("SQL") or row.get("query")
        question = row.get("question")
        evidence = row.get("evidence", "")
        db_path = _db_path(config, db_id)

        gold_result = execute_sql(db_path, gold_sql, timeout=config.error_mining.execution_timeout)
        if gold_result.result_rows is None:
            continue

        schema = get_schema_service().load_sqlite_schema(db_path)
        prompt = format_sql_generation_prompt(schema, question, evidence)
        try:
            responses, _usage = generation_llm.ask([{"role": "user", "content": prompt}], n=1, temperature=0.7, timeout=90)
            cand_sql = parse_sql_result(responses[0].content)
        except Exception as e:
            logger.warning(f"[rule_validation] correct-candidate generation failed for train idx={idx}: {e}")
            continue
        if not cand_sql:
            continue

        cand_result = execute_sql(db_path, cand_sql, timeout=config.error_mining.execution_timeout)
        if _rows_equal(cand_result, gold_result):
            correct_half.append({
                "question_id": idx, "db_id": db_id, "question": question,
                "evidence": evidence, "gold_sql": gold_sql, "candidate_sql": cand_sql,
            })

    if len(correct_half) < n_target:
        logger.warning(
            f"[rule_validation] only found {len(correct_half)}/{n_target} correct candidates "
            f"after scanning {len(order)} training questions"
        )
    return correct_half


def build_validation_batch(config: RuleCreatorConfig, generation_llm) -> List[Dict[str, Any]]:
    """One shared mixed batch, drawn once and reused across every rule and
    every refinement round (see module docstring)."""
    cfg = config.rule_validation
    n_incorrect = cfg.batch_size // 2
    n_correct = cfg.batch_size - n_incorrect

    fail_final = _load_fail_final(config.fail_final_path)
    rng = random.Random(0)
    rng.shuffle(fail_final)
    seen_sql: set = set()
    incorrect_half: List[Dict[str, Any]] = []
    for row in fail_final:
        norm = _normalize_sql(row["candidate_sql"])
        if norm in seen_sql:
            continue
        seen_sql.add(norm)
        incorrect_half.append(row)
        if len(incorrect_half) >= n_incorrect:
            break

    correct_half = _build_correct_candidate_half(config, generation_llm, n_correct, seed=1)

    batch = [{**r, "is_correct": False} for r in incorrect_half] + [{**r, "is_correct": True} for r in correct_half]
    logger.info(f"[rule_validation] mixed batch: {len(incorrect_half)} incorrect + {len(correct_half)} correct = {len(batch)}")
    return batch


def _find_group_for_rule(rule: Dict[str, Any], error_groups_path: str) -> Dict[str, Any]:
    """Maps a synthesized rule back to the clustering group it was synthesized
    from, via the group label embedded verbatim in the rule's provenance note
    (see rule_synthesis.py's note-writing code) -- so a targeted validation
    batch can draw 'incorrect' candidates that are actually on-topic for this
    rule, instead of build_validation_batch's blind random draw from the full
    fail_final pool. That blind draw hits this rule's own error type at
    roughly support/n_explanations_in rate -- ~1-2% for a typical rule here --
    so a small random batch very likely contains zero relevant cases even
    when the rule itself is sound (confirmed directly: a 12-item random batch
    gave RC-PROJECTION-EXTRA-COLUMNS and RC-EXTRA-RESULT-COLUMNS 0 helped/0
    harmed each, 2026-10-01 probe run, job 1361976)."""
    note = rule.get("note", "")
    m = re.search(r"dominant error group \('([^']+)'\)", note)
    if not m:
        raise ValueError(f"could not find a group label in the note for {rule['rule_name']!r}: {note[:200]!r}")
    label = m.group(1)
    groups = json.load(open(error_groups_path))["groups"]
    matches = [g for g in groups if g["label"] == label]
    if not matches:
        raise ValueError(f"no error group labeled {label!r} found in {error_groups_path}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} groups share label {label!r} in {error_groups_path} -- ambiguous")
    return matches[0]


def build_targeted_validation_batch(
    config: RuleCreatorConfig,
    rule: Dict[str, Any],
    generation_llm,
    n_incorrect: int,
    n_correct: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Per-rule validation batch: the 'incorrect' half is drawn from the
    rule's OWN clustering group (04_error_groups.json's member_rows) instead
    of build_validation_batch's blind random sample of the whole fail_final
    pool, so helped/harmed counts reflect the rule's actual behavior rather
    than batch-composition luck. The 'correct' half is still a fresh random
    draw (same as build_validation_batch) since harm-testing is
    intentionally rule-agnostic -- a rule should not fire on arbitrary
    already-correct SQL regardless of what it was mined from."""
    group = _find_group_for_rule(rule, config.error_groups_path)
    members = list(group["member_rows"])
    rng = random.Random(seed)
    rng.shuffle(members)

    seen_sql: set = set()
    incorrect_half: List[Dict[str, Any]] = []
    for row in members:
        norm = _normalize_sql(row["candidate_sql"])
        if norm in seen_sql:
            continue
        seen_sql.add(norm)
        incorrect_half.append(row)
        if len(incorrect_half) >= n_incorrect:
            break

    correct_half = _build_correct_candidate_half(config, generation_llm, n_correct, seed=seed + 1)

    batch = [{**r, "is_correct": False} for r in incorrect_half] + [{**r, "is_correct": True} for r in correct_half]
    logger.info(
        f"[rule_validation] targeted batch for {rule['rule_name']} (group_id={group['group_id']}, "
        f"label={group['label']!r}): {len(incorrect_half)}/{len(members)} available incorrect "
        f"+ {len(correct_half)} correct = {len(batch)}"
    )
    return batch


def _apply_rule_to_item(rule: Dict[str, Any], item: Dict[str, Any], correction_llm, db_path: str) -> Optional[str]:
    """Standalone rule-relevance-selection + correction call, reusing
    dextersql/rule_correction/prompts.py::format_rule_correction_prompt with
    a single-rule list instead of SQLRuleCorrectionRunner's full rule set and
    full-pipeline DataItem. Returns corrected_sql if the rule applied and
    changed the query, or None if the rule legitimately did not apply
    (well-formed response, rule_name absent from applied_rules) -- only a
    malformed <result> block is retried, up to _MAX_FORMAT_RETRY times."""
    schema = get_schema_service().load_sqlite_schema(db_path)
    prompt = format_rule_correction_prompt(
        database_schema=schema,
        question=item["question"],
        evidence=item.get("evidence", ""),
        sql=item["candidate_sql"],
        rules=[rule],
    )
    for attempt in range(_MAX_FORMAT_RETRY):
        try:
            responses, _usage = correction_llm.ask([{"role": "user", "content": prompt}], n=1, timeout=90)  # temperature comes from config.rule_validation.llm -- do not hardcode it here, it overrides the config value
            shape = parse_rule_correction_shape(responses[0].content)
        except Exception as e:
            logger.warning(f"[rule_validation] correction call failed (attempt {attempt + 1}): {e}")
            shape = None
        if shape is not None:
            if rule["rule_name"] not in shape["applied_rules"] or not shape["corrected_sql"]:
                return None  # legitimately not applicable -- do not retry
            return shape["corrected_sql"]
    logger.warning(f"[rule_validation] gave up on malformed correction response for rule={rule['rule_name']}")
    return None


def _classify_case(item: Dict[str, Any], corrected_sql: Optional[str], config: RuleCreatorConfig) -> Tuple[str, str]:
    """Returns (outcome, after_sql) where outcome is one of helped/harmed/unchanged."""
    before_sql = item["candidate_sql"]
    if corrected_sql is None:
        return "unchanged", before_sql

    db_path = _db_path(config, item["db_id"])
    gold_result = execute_sql(db_path, item["gold_sql"], timeout=config.error_mining.execution_timeout)
    after_result = execute_sql(db_path, corrected_sql, timeout=config.error_mining.execution_timeout)
    after_correct = _rows_equal(after_result, gold_result)

    if item["is_correct"] and not after_correct:
        return "harmed", corrected_sql
    if not item["is_correct"] and after_correct:
        return "helped", corrected_sql
    return "unchanged", corrected_sql


def _validate_once(rule: Dict[str, Any], batch: List[Dict[str, Any]], correction_llm, config: RuleCreatorConfig) -> Dict[str, List[Dict[str, Any]]]:
    cases: Dict[str, List[Dict[str, Any]]] = {"helped": [], "harmed": [], "unchanged": []}
    for item in batch:
        db_path = _db_path(config, item["db_id"])
        corrected_sql = _apply_rule_to_item(rule, item, correction_llm, db_path)
        outcome, after_sql = _classify_case(item, corrected_sql, config)
        cases[outcome].append({
            "question": item["question"], "gold_sql": item["gold_sql"],
            "before_sql": item["candidate_sql"], "after_sql": after_sql, "outcome": outcome,
        })
    return cases


def _passes(cases: Dict[str, List[Dict[str, Any]]], threshold: float) -> bool:
    n_helped = len(cases["helped"])
    n_harmed = len(cases["harmed"])
    return n_helped >= 1 and (n_harmed / n_helped) < threshold


def run_rule_validation(config: RuleCreatorConfig, rules: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if rules is None:
        rules = _load_created_rules(config.created_rules_path)
    print(f"[rule_validation] validating {len(rules)} synthesized rules")

    configure_schema_service()
    correction_llm = make_llm(config.rule_validation.llm)
    generation_llm = make_llm(config.error_mining.generation_llm)

    batch = build_validation_batch(config, generation_llm)

    accepted: List[Dict[str, Any]] = []
    for rule in rules:
        current_rule = dict(rule)
        for round_idx in range(config.rule_validation.max_refine_rounds + 1):
            cases = _validate_once(current_rule, batch, correction_llm, config)
            n_helped, n_harmed = len(cases["helped"]), len(cases["harmed"])
            harm_ratio = (n_harmed / n_helped) if n_helped else float("inf")
            print(
                f"[rule_validation]   {current_rule['rule_name']} round={round_idx} "
                f"helped={n_helped} harmed={n_harmed} harm_ratio={harm_ratio:.2f}"
            )
            if _passes(cases, config.rule_validation.harm_ratio_threshold):
                accepted.append(current_rule)
                print(f"[rule_validation]   {current_rule['rule_name']} ACCEPTED (round {round_idx})")
                break
            if round_idx == config.rule_validation.max_refine_rounds:
                print(f"[rule_validation]   {current_rule['rule_name']} DISCARDED after {round_idx} refine rounds")
                break

            refine_prompt = format_rule_refinement_prompt(current_rule, cases["helped"], cases["harmed"], cases["unchanged"])
            try:
                responses, _usage = correction_llm.ask([{"role": "user", "content": refine_prompt}], n=1, temperature=0.7, timeout=90)
                refined = parse_rule_refinement(responses[0].content)
            except Exception as e:
                logger.warning(f"[rule_validation] refinement call failed for {current_rule['rule_name']}: {e}")
                refined = None
            if refined is None:
                print(f"[rule_validation]   {current_rule['rule_name']} refinement call failed, stopping early -> discarded")
                break
            refined["rule_name"] = current_rule["rule_name"]  # name is pinned across refinement rounds
            current_rule = refined

    out_path = Path(config.validated_rules_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)

    print(f"[rule_validation] {len(accepted)}/{len(rules)} rules accepted -> {out_path}")
    return accepted


def _validate_one_rule_targeted(rule: Dict[str, Any], config: RuleCreatorConfig, correction_llm, generation_llm) -> Dict[str, Any]:
    """One rule's full targeted-validation lifecycle (batch build -> validate
    -> refine up to max_refine_rounds). Pure function of (rule, config) plus
    the two shared LLM clients -- safe to run concurrently across rules in a
    ThreadPoolExecutor, since build_targeted_validation_batch/_validate_once/
    execute_sql are all already designed for concurrent use (see module
    docstring's note on the shared _SQL_EXECUTOR pool)."""
    name = rule["rule_name"]
    cfg = config.rule_validation
    try:
        group = _find_group_for_rule(rule, config.error_groups_path)
    except Exception as e:
        return {"rule_name": name, "accepted": False, "final_rule": None, "error": f"group lookup failed: {e}", "rounds": []}

    try:
        batch = build_targeted_validation_batch(config, rule, generation_llm, cfg.n_incorrect, cfg.n_correct, seed=0)
    except Exception as e:
        return {"rule_name": name, "accepted": False, "final_rule": None, "error": f"batch build failed: {e}",
                "group_id": group["group_id"], "group_label": group["label"], "rounds": []}

    current_rule = dict(rule)
    round_logs: List[Dict[str, Any]] = []
    accepted = False
    for round_idx in range(cfg.max_refine_rounds + 1):
        cases = _validate_once(current_rule, batch, correction_llm, config)
        n_helped, n_harmed = len(cases["helped"]), len(cases["harmed"])
        harm_ratio = (n_harmed / n_helped) if n_helped else None
        round_logs.append({
            "round": round_idx, "helped": n_helped, "harmed": n_harmed,
            "unchanged": len(cases["unchanged"]), "harm_ratio": harm_ratio,
            "helped_examples": cases["helped"][:3], "harmed_examples": cases["harmed"][:3],
        })
        print(f"[rule_validation]   {name} round={round_idx} helped={n_helped} harmed={n_harmed} "
              f"harm_ratio={'inf' if harm_ratio is None else f'{harm_ratio:.2f}'}")

        if _passes(cases, cfg.harm_ratio_threshold):
            accepted = True
            print(f"[rule_validation]   {name} ACCEPTED (round {round_idx})")
            break
        if round_idx == cfg.max_refine_rounds:
            print(f"[rule_validation]   {name} DISCARDED after {round_idx} refine rounds")
            break

        refine_prompt = format_rule_refinement_prompt(current_rule, cases["helped"], cases["harmed"], cases["unchanged"])
        try:
            responses, _usage = correction_llm.ask([{"role": "user", "content": refine_prompt}], n=1, temperature=0.7, timeout=90)
            refined = parse_rule_refinement(responses[0].content)
        except Exception as e:
            logger.warning(f"[rule_validation] refinement call failed for {name}: {e}")
            refined = None
        if refined is None:
            print(f"[rule_validation]   {name} refinement call failed to parse, stopping early -> discarded")
            break
        refined["rule_name"] = name
        current_rule = refined

    return {
        "rule_name": name, "accepted": accepted, "final_rule": current_rule if accepted else None,
        "group_id": group["group_id"], "group_label": group["label"],
        "n_cluster_members": len(group["member_rows"]), "batch_size": len(batch),
        "rounds": round_logs,
    }


def run_targeted_rule_validation(
    config: RuleCreatorConfig,
    rules: List[Dict[str, Any]] = None,
    seed_accepted: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """The recommended Step 5 entry point (config.rule_validation.sampling_strategy
    == "targeted", the default): validates every rule against a batch drawn from
    its OWN clustering group instead of a blind random draw (see
    RuleValidationConfig's docstring for why this matters), in parallel across
    rules (config.rule_validation.n_parallel workers, same pattern as
    error_mining.run_error_mining), with every rule's outcome appended to
    config.validation_checkpoint_path immediately -- so a killed/restarted run
    only re-validates what's left, and partial progress is never lost (fixing
    the gap flagged during development: the legacy run_rule_validation only
    wrote its output once, at the very end).

    seed_accepted: optional {rule_name: rule_dict} already known-accepted from
    a prior run (e.g. a hand-picked probe) -- written straight to the
    checkpoint as accepted, without re-validating, same as
    run_rule_validation's `rules` parameter lets a caller pass a subset.
    """
    if rules is None:
        rules = _load_created_rules(config.created_rules_path)
    rules_by_name = {r["rule_name"]: r for r in rules}

    checkpoint_path = Path(config.validation_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_done.add(json.loads(line)["rule_name"])

    todo = [n for n in rules_by_name if n not in already_done]
    print(f"[rule_validation] {len(rules_by_name)} targeted, {len(already_done)} already done, {len(todo)} to process")
    print(f"[rule_validation] n_incorrect={config.rule_validation.n_incorrect} n_correct={config.rule_validation.n_correct} "
          f"n_parallel={config.rule_validation.n_parallel} harm_ratio_threshold={config.rule_validation.harm_ratio_threshold} "
          f"max_refine_rounds={config.rule_validation.max_refine_rounds}")

    configure_schema_service()
    correction_llm = make_llm(config.rule_validation.llm)
    generation_llm = make_llm(config.error_mining.generation_llm)

    checkpoint_f = open(checkpoint_path, "a")

    def _write_result(result: Dict[str, Any]) -> None:
        checkpoint_f.write(json.dumps(result) + "\n")
        checkpoint_f.flush()

    for name, rule in (seed_accepted or {}).items():
        if name in already_done:
            continue
        _write_result({"rule_name": name, "accepted": True, "final_rule": rule, "source": "seeded", "rounds": []})
        already_done.add(name)
        if name in todo:
            todo.remove(name)
        print(f"[rule_validation] seeded {name} as already-accepted")

    completed = 0
    with ThreadPoolExecutor(max_workers=config.rule_validation.n_parallel) as pool:
        futures = {pool.submit(_validate_one_rule_targeted, rules_by_name[name], config, correction_llm, generation_llm): name for name in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.warning(f"[rule_validation] {name} raised unexpectedly: {e}")
                result = {"rule_name": name, "accepted": False, "final_rule": None, "error": str(e), "rounds": []}
            _write_result(result)
            completed += 1
            if completed % 5 == 0 or completed == len(todo):
                print(f"[rule_validation] {completed}/{len(todo)} rules processed")

    checkpoint_f.close()

    all_results = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    accepted = [r["final_rule"] for r in all_results if r.get("accepted") and r.get("final_rule")]
    out_path = Path(config.validated_rules_path)
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)

    print(f"[rule_validation] {len(accepted)}/{len(all_results)} rules accepted overall -> {out_path}")
    return accepted
