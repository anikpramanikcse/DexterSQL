"""
Turns a chosen subset of validated rules into a dextersql.rule_correction.rules
-shaped Python module (RULES/RULE_NAMES, CorrectionRule TypedDict) -- the
format dextersql/rule_correction/sql_rule_correction.py imports its active
rule set from.

This is the gap named explicitly during development: validation (Step 5)
tells you WHICH rules pass, but turning "these N accepted rules" into "the
production rule set" is still a judgment call (dedup near-identical rules,
match against/compare to existing canonical rules, decide what's in scope)
-- not automated here, and not meant to be. What this script DOES automate
is the mechanical last step once that judgment call is made: given a final
list of rule_names, produce a clean, importable module, instead of
hand-copy-pasting rule dicts into a script (as the first mined-3-rules
experiment did before this was written).

Usage:
    python -m dextersql.rule_creator.export_rules \\
        --validated-rules results_ruleval_3000/06_validated_rules.json \\
        --rule-names RC-PROJECTION-EXTRA-COLUMNS RC-COUNT-DISTINCT-MISUSE RC-MINMAX-EQ-ORDER-LIMIT \\
        --out generated_rules/mined_3rules.py

The output module is a drop-in for dextersql.rule_correction.rules -- either
point PYTHONPATH/sys.path tricks at it, or (preferred, since it never edits
any shared file on disk) monkeypatch dextersql.rule_correction.rules.RULES
from it at runtime, which is exactly what scripts/run_rule_set_experiment.py
does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

_MODULE_TEMPLATE = '''"""
Correction rules for the sql_rule_correction stage -- GENERATED FILE.

Exported by dextersql.rule_creator.export_rules from {source!r}.
Rule names: {rule_names!r}
{extra_header}
Do not hand-edit -- regenerate from the source validated-rules file instead,
so provenance (which Rule Creator run, which validation pass) stays accurate.
"""

from typing import List, TypedDict


class CorrectionRule(TypedDict):
    rule_name: str
    gist: str
    bad_pattern: str
    correct_pattern: str
    fix: str
    note: str  # real-example provenance; for human debugging only, never sent to the LLM


RULES: List[CorrectionRule] = {rules_repr}

RULE_NAMES: List[str] = [r["rule_name"] for r in RULES]
'''


def export_as_correction_rules_module(
    rules: List[Dict[str, Any]],
    out_path: str,
    source: str = "",
    extra_header: str = "",
) -> None:
    """Writes `rules` (each a dict with rule_name/gist/bad_pattern/correct_pattern/fix,
    optionally note) as a rules.py-shaped module at out_path. `rules` order is
    preserved as RULES' order."""
    required_keys = {"rule_name", "gist", "bad_pattern", "correct_pattern", "fix"}
    for r in rules:
        missing = required_keys - set(r.keys())
        if missing:
            raise ValueError(f"rule {r.get('rule_name', '<unknown>')!r} is missing required fields: {missing}")

    clean_rules = [
        {
            "rule_name": r["rule_name"],
            "gist": r["gist"],
            "bad_pattern": r["bad_pattern"],
            "correct_pattern": r["correct_pattern"],
            "fix": r["fix"],
            "note": r.get("note", ""),
        }
        for r in rules
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = _MODULE_TEMPLATE.format(
        source=source,
        rule_names=[r["rule_name"] for r in clean_rules],
        extra_header=("\n" + extra_header.strip() + "\n") if extra_header.strip() else "",
        rules_repr=_pretty_list_repr(clean_rules),
    )
    out.write_text(content)
    print(f"[export_rules] wrote {len(clean_rules)} rules -> {out}")


def _pretty_list_repr(rules: List[Dict[str, str]], indent: int = 4) -> str:
    """repr() a list of flat str-valued dicts with stable key order and
    readable multi-line formatting (black-ish), since the stdlib pprint
    module doesn't preserve insertion order the way we want for diffing."""
    pad = " " * indent
    lines = ["["]
    for r in rules:
        lines.append(pad + "{")
        for key in ("rule_name", "gist", "bad_pattern", "correct_pattern", "fix", "note"):
            lines.append(pad + "    " + json.dumps(key) + ": " + json.dumps(r[key]) + ",")
        lines.append(pad + "},")
    lines.append("]")
    return "\n".join(lines)


def load_rules_by_name(validated_rules_path: str) -> Dict[str, Dict[str, Any]]:
    with open(validated_rules_path) as f:
        rules = json.load(f)
    return {r["rule_name"]: r for r in rules}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validated-rules", required=True, help="Path to a validated-rules JSON file (e.g. 06_validated_rules.json)")
    parser.add_argument("--rule-names", nargs="+", required=True, help="Which rule_names to include, in this order")
    parser.add_argument("--out", required=True, help="Output path for the generated rules module (e.g. generated_rules/mined_3rules.py)")
    parser.add_argument("--note", default="", help="Optional extra line(s) for the generated module's header docstring")
    args = parser.parse_args()

    by_name = load_rules_by_name(args.validated_rules)
    missing = [n for n in args.rule_names if n not in by_name]
    if missing:
        raise SystemExit(f"rule_names not found in {args.validated_rules}: {missing}")

    selected = [by_name[n] for n in args.rule_names]
    export_as_correction_rules_module(selected, args.out, source=args.validated_rules, extra_header=args.note)


if __name__ == "__main__":
    main()
