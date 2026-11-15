"""
Shared <result> parsing helpers for the Rule Creator pipeline.

Every function here has the (content: str, **kwargs) -> Optional[T] shape
that dextersql.core.llm_extractor.extractor.LLMExtractor.extract_with_retry
expects as its `rule_parser` argument -- returning None signals "invalid,
retry" to the extractor.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from dextersql.core.logger import logger

_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)


def _extract_result_block(content: str) -> Optional[str]:
    m = _RESULT_RE.search(content)
    if not m:
        return None
    raw = m.group(1).strip()
    raw = re.sub(r"^```(?:json|sql)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_json_result(content: str) -> Optional[Any]:
    raw = _extract_result_block(content)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.debug(f"[rule_creator] JSON parse failed on <result>: {raw[:300]}")
        return None


def parse_sql_result(content: str) -> Optional[str]:
    """Step 1a: <result>SQL</result> -- bare SQL text, not JSON."""
    raw = _extract_result_block(content)
    if not raw:
        return None
    return raw if raw.strip() else None


def parse_explanation_list(content: str) -> Optional[List[str]]:
    """Step 1b: <result>["explanation 1", "explanation 2", ...]</result>. An
    empty list is a *valid* result (the model found no genuine SQL-formulation
    difference), so we distinguish "parse failed" (None) from "parsed to []"."""
    obj = _parse_json_result(content)
    if obj is None or not isinstance(obj, list):
        return None
    return [str(x).strip() for x in obj if str(x).strip()]


def parse_db_agnostic_classification(content: str) -> Optional[str]:
    """Step 2: <result>{"classification": "database_agnostic"|"database_specific"}</result>."""
    obj = _parse_json_result(content)
    if not isinstance(obj, dict):
        return None
    cls = obj.get("classification")
    if cls not in ("database_agnostic", "database_specific"):
        return None
    return cls


def parse_batch_clustering(content: str, n: int) -> Optional[List[Dict[str, Any]]]:
    """Step 3 (batch): <result>[{"label": str, "member_ids": [int, ...]}, ...]</result>.
    Validates every id in 1..n is covered exactly once; on any violation
    returns None so extract_with_retry retries rather than silently dropping
    or duplicating an explanation into two groups."""
    obj = _parse_json_result(content)
    if not isinstance(obj, list) or not obj:
        return None

    seen: Dict[int, int] = {}
    groups: List[Dict[str, Any]] = []
    for g in obj:
        if not isinstance(g, dict) or "label" not in g or "member_ids" not in g:
            return None
        member_ids = g["member_ids"]
        if not isinstance(member_ids, list) or not member_ids:
            return None
        try:
            member_ids = [int(x) for x in member_ids]
        except (TypeError, ValueError):
            return None
        for mid in member_ids:
            seen[mid] = seen.get(mid, 0) + 1
        groups.append({"label": str(g["label"]).strip(), "member_ids": member_ids})

    if set(seen.keys()) != set(range(1, n + 1)) or any(c != 1 for c in seen.values()):
        logger.debug(f"[rule_creator] batch clustering coverage mismatch: expected 1..{n} each once, got {seen}")
        return None
    return groups


def parse_group_merge(content: str, n: int) -> Optional[List[Dict[str, Any]]]:
    """Step 3 (merge): <result>[{"merged_label": str, "source_group_ids": [int, ...]}, ...]</result>.
    Same coverage validation as parse_batch_clustering."""
    obj = _parse_json_result(content)
    if not isinstance(obj, list) or not obj:
        return None

    seen: Dict[int, int] = {}
    merges: List[Dict[str, Any]] = []
    for m in obj:
        if not isinstance(m, dict) or "merged_label" not in m or "source_group_ids" not in m:
            return None
        source_ids = m["source_group_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            return None
        try:
            source_ids = [int(x) for x in source_ids]
        except (TypeError, ValueError):
            return None
        for sid in source_ids:
            seen[sid] = seen.get(sid, 0) + 1
        merges.append({"merged_label": str(m["merged_label"]).strip(), "source_group_ids": source_ids})

    if set(seen.keys()) != set(range(1, n + 1)) or any(c != 1 for c in seen.values()):
        logger.debug(f"[rule_creator] group merge coverage mismatch: expected 1..{n} each once, got {seen}")
        return None
    return merges


_RULE_NAME_RE = re.compile(r"^RC-[A-Z0-9-]+$")
_REQUIRED_RULE_FIELDS = ("rule_name", "gist", "bad_pattern", "correct_pattern", "fix")


def _parse_rule_shaped_result(content: str) -> Optional[Dict[str, str]]:
    obj = _parse_json_result(content)
    if not isinstance(obj, dict):
        return None
    if not all(k in obj and str(obj[k]).strip() for k in _REQUIRED_RULE_FIELDS):
        return None
    rule_name = str(obj["rule_name"]).strip().upper()
    if not _RULE_NAME_RE.match(rule_name):
        return None
    return {k: str(obj[k]).strip() for k in _REQUIRED_RULE_FIELDS} | {"rule_name": rule_name}


def parse_rule_synthesis(content: str) -> Optional[Dict[str, str]]:
    """Step 4: <result>{"rule_name", "gist", "bad_pattern", "correct_pattern", "fix"}</result>."""
    return _parse_rule_shaped_result(content)


def parse_rule_refinement(content: str) -> Optional[Dict[str, str]]:
    """Step 5: <result>{"rule_name", "gist", "bad_pattern", "correct_pattern", "fix"}</result> --
    same shape and validation as parse_rule_synthesis; kept as a separate
    named function since it's a conceptually distinct call site (refinement,
    not first-time synthesis) even though the wire format is identical."""
    return _parse_rule_shaped_result(content)


def parse_rule_correction_shape(content: str) -> Optional[Dict[str, Any]]:
    """Step 5: validates ONLY that <result>{"applied_rules": [...], "corrected_sql": "..."}</result>
    is well-formed -- does not check whether any particular rule_name is in
    applied_rules. "The rule doesn't apply here" (applied_rules=[]) is a
    valid, well-formed response, not a parse failure, so format validity and
    "did this specific rule apply" are checked separately: retry a malformed
    response, but never retry a legitimately negative one."""
    obj = _parse_json_result(content)
    if not isinstance(obj, dict):
        return None
    applied_rules = obj.get("applied_rules")
    corrected_sql = obj.get("corrected_sql")
    if not isinstance(applied_rules, list) or not isinstance(corrected_sql, str):
        return None
    return {"applied_rules": applied_rules, "corrected_sql": corrected_sql.strip()}


def parse_rule_correction_result(content: str, rule_name: str) -> Optional[str]:
    """Step 5: same response as parse_rule_correction_shape, but returns the
    corrected SQL only if rule_name appears in applied_rules and
    corrected_sql is non-empty -- None otherwise (either malformed, or the
    rule legitimately did not apply to this candidate). Callers that need to
    tell "malformed" apart from "did not apply" should call
    parse_rule_correction_shape first."""
    shape = parse_rule_correction_shape(content)
    if shape is None:
        return None
    if rule_name not in shape["applied_rules"] or not shape["corrected_sql"]:
        return None
    return shape["corrected_sql"]
