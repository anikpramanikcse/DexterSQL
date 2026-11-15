"""
Configuration for the Rule Creator pipeline.

Deliberately independent of dextersql.core.config.AppConfig, which is built
around DatasetConfig(split="dev"|"test") for the inference-time pipeline.
Rule Creator only ever reads BIRD *training* data, and giving it its own
config -- with explicit train_json_path / train_databases_root fields and no
concept of "split" at all -- makes that isolation structural rather than a
matter of remembering to pass the right flag.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from dextersql.core.config.config import LLMConfig


class SamplingConfig(BaseModel):
    """Step 1a: stratified sampling of training questions by difficulty."""

    target_size: int = Field(default=3000, description="Total number of training questions to sample")
    seed: int = Field(default=7)
    simple_fraction: float = Field(default=0.40)
    moderate_fraction: float = Field(default=0.35)
    challenging_fraction: float = Field(default=0.25)

    @model_validator(mode="after")
    def _check_fractions(self):
        total = self.simple_fraction + self.moderate_fraction + self.challenging_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sampling fractions must sum to 1.0, got {total}")
        return self


class ErrorMiningConfig(BaseModel):
    """Step 1: candidate generation, execution comparison, failure explanation."""

    candidates_per_question: int = Field(default=4, description="Candidate SQLs sampled per question")
    n_parallel: int = Field(default=4, description="Questions processed concurrently")
    execution_timeout: int = Field(default=30, description="Seconds before a SQL execution against a train DB is aborted")
    generation_llm: LLMConfig
    explanation_llm: LLMConfig


class DbAgnosticFilterConfig(BaseModel):
    """Step 2: keep only reusable, database-agnostic failure explanations."""

    n_parallel: int = Field(default=8)
    llm: LLMConfig


class ClusteringConfig(BaseModel):
    """Step 3: hierarchical clustering of database-agnostic explanations."""

    batch_size: int = Field(default=15, description="Explanations per batch-level clustering call, within one database")
    cross_db_merge_chunk_size: int = Field(
        default=20,
        description="Groups per merge call when merging across databases -- keeps the merge call's token budget from being overwhelmed (see README.md, lesson learned the hard way on a different clustering pass)",
    )
    min_group_support: int = Field(default=3, description="Groups with fewer supporting explanations than this are discarded as non-dominant")
    llm: LLMConfig


class RuleSynthesisConfig(BaseModel):
    """Step 4: convert each dominant error group into a canonical rule."""

    max_examples_per_rule: int = Field(default=6, description="Representative member explanations shown to the LLM per group, to keep the synthesis prompt bounded")
    llm: LLMConfig


class RuleValidationConfig(BaseModel):
    """Step 5: validate each synthesized rule against a mixed batch of
    correct/incorrect training candidates, and refine it if it doesn't
    clear the harm-ratio threshold. Second half of the paper's Section
    3.1.4 Step 4 (rule synthesis + validation are one paper step, two
    pipeline steps here, matching the pattern of Step 2/Step 3 -- see
    pipeline.py).

    sampling_strategy="targeted" (the default) draws each rule's "incorrect"
    half from that rule's OWN clustering group in 04_error_groups.json
    (rule_validation.build_targeted_validation_batch), instead of a blind
    random draw from the whole fail_final pool. This matters: a rule mined
    from a cluster of size 74 out of a ~4500-explanation pool has roughly a
    1.5% base rate, so a small random batch (sampling_strategy="random",
    batch_size items) has a high chance of containing zero matching cases
    purely by chance -- confirmed directly on 2026-10-01 (two strong
    candidate rules scored 0 helped/0 harmed on a random 12-item batch, then
    helped=2 and helped=6 with 0 harmed once validated against their own
    cluster). "random" is kept only for comparison/regression testing."""

    sampling_strategy: str = Field(default="targeted", description="'targeted' (recommended) draws the incorrect half from the rule's own cluster; 'random' is the legacy blind full-pool draw")

    # -- "random" strategy only (run_rule_validation) --
    batch_size: int = Field(default=20, description="[random strategy] mixed correct+incorrect candidates sampled once from the whole pool, reused across every rule and every refinement round")

    # -- "targeted" strategy only (run_targeted_rule_validation) --
    n_incorrect: int = Field(default=30, description="[targeted strategy] incorrect candidates drawn from the rule's own cluster, capped at the cluster's actual size")
    n_correct: int = Field(default=16, description="[targeted strategy] random correct candidates (not from the rule's cluster) -- a rule-agnostic test for false positives/harm")
    n_parallel: int = Field(default=12, description="[targeted strategy] rules validated concurrently (ThreadPoolExecutor), same pattern as error_mining.n_parallel")

    harm_ratio_threshold: float = Field(default=0.2, description="Rule is rejected if |harmed| / |helped| >= this value")
    max_refine_rounds: int = Field(default=3, description="Refinement attempts before a rule is discarded")
    llm: LLMConfig


class RuleCreatorConfig(BaseModel):
    # -- The only paths this pipeline ever reads. There is deliberately no
    # "split" concept and no path here that could resolve to dev/test data. --
    train_json_path: str = Field(..., description="Path to BIRD train.json (question/evidence/SQL/db_id per row)")
    train_databases_root: str = Field(..., description="Path to train_databases/ (one subfolder per db_id, each holding <db_id>.sqlite)")

    output_dir: str = Field(..., description="Where every step writes its checkpoint/output files")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    error_mining: ErrorMiningConfig
    db_agnostic_filter: DbAgnosticFilterConfig
    clustering: ClusteringConfig
    rule_synthesis: RuleSynthesisConfig
    rule_validation: RuleValidationConfig

    @model_validator(mode="after")
    def _check_train_paths_exist(self):
        if not Path(self.train_json_path).is_file():
            raise ValueError(f"train_json_path does not exist: {self.train_json_path}")
        if not Path(self.train_databases_root).is_dir():
            raise ValueError(f"train_databases_root is not a directory: {self.train_databases_root}")
        return self

    # -- output file paths, one per pipeline step, all under output_dir --
    @property
    def sampled_path(self) -> str:
        return str(Path(self.output_dir) / "01_sampled_questions.json")

    @property
    def fail_candidate_path(self) -> str:
        return str(Path(self.output_dir) / "02_fail_candidate.jsonl")

    @property
    def fail_final_path(self) -> str:
        return str(Path(self.output_dir) / "03_fail_final.jsonl")

    @property
    def error_groups_path(self) -> str:
        return str(Path(self.output_dir) / "04_error_groups.json")

    @property
    def created_rules_path(self) -> str:
        return str(Path(self.output_dir) / "05_created_rules.json")

    @property
    def validated_rules_path(self) -> str:
        return str(Path(self.output_dir) / "06_validated_rules.json")

    @property
    def validation_checkpoint_path(self) -> str:
        """Incremental, resumable per-rule results for run_targeted_rule_validation
        -- one JSON line per rule as soon as it's decided, keyed by rule_name, so a
        killed/restarted run only re-does what's left (same pattern as
        error_mining's fail_candidate_path)."""
        return str(Path(self.output_dir) / "06_validation_checkpoint.jsonl")


def load_rule_creator_config(path: str) -> RuleCreatorConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    def llm(section_name: str, sub_key: str) -> LLMConfig:
        return LLMConfig(**raw[section_name][sub_key])

    sampling = SamplingConfig(**raw.get("sampling", {}))

    error_mining_raw = raw["error_mining"]
    error_mining = ErrorMiningConfig(
        candidates_per_question=error_mining_raw.get("candidates_per_question", 4),
        n_parallel=error_mining_raw.get("n_parallel", 4),
        execution_timeout=error_mining_raw.get("execution_timeout", 30),
        generation_llm=LLMConfig(**error_mining_raw["generation_llm"]),
        explanation_llm=LLMConfig(**error_mining_raw["explanation_llm"]),
    )

    db_agnostic_raw = raw["db_agnostic_filter"]
    db_agnostic_filter = DbAgnosticFilterConfig(
        n_parallel=db_agnostic_raw.get("n_parallel", 8),
        llm=LLMConfig(**db_agnostic_raw["llm"]),
    )

    clustering_raw = raw["clustering"]
    clustering = ClusteringConfig(
        batch_size=clustering_raw.get("batch_size", 15),
        cross_db_merge_chunk_size=clustering_raw.get("cross_db_merge_chunk_size", 20),
        min_group_support=clustering_raw.get("min_group_support", 3),
        llm=LLMConfig(**clustering_raw["llm"]),
    )

    rule_synthesis_raw = raw["rule_synthesis"]
    rule_synthesis = RuleSynthesisConfig(
        max_examples_per_rule=rule_synthesis_raw.get("max_examples_per_rule", 6),
        llm=LLMConfig(**rule_synthesis_raw["llm"]),
    )

    rule_validation_raw = raw["rule_validation"]
    rule_validation = RuleValidationConfig(
        sampling_strategy=rule_validation_raw.get("sampling_strategy", "targeted"),
        batch_size=rule_validation_raw.get("batch_size", 20),
        n_incorrect=rule_validation_raw.get("n_incorrect", 30),
        n_correct=rule_validation_raw.get("n_correct", 16),
        n_parallel=rule_validation_raw.get("n_parallel", 12),
        harm_ratio_threshold=rule_validation_raw.get("harm_ratio_threshold", 0.2),
        max_refine_rounds=rule_validation_raw.get("max_refine_rounds", 3),
        llm=LLMConfig(**rule_validation_raw["llm"]),
    )

    return RuleCreatorConfig(
        train_json_path=raw["data"]["train_json_path"],
        train_databases_root=raw["data"]["train_databases_root"],
        output_dir=raw["data"]["output_dir"],
        sampling=sampling,
        error_mining=error_mining,
        db_agnostic_filter=db_agnostic_filter,
        clustering=clustering,
        rule_synthesis=rule_synthesis,
        rule_validation=rule_validation,
    )
