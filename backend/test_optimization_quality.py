"""
Phase D3: Optimization Quality Benchmark Suite

Evaluates whether prompt transformations preserve quality across six dimensions:
1. Intent preservation (core task and objective remain intact)
2. Requirement preservation (deliverables and functionality retained)
3. Constraint / negation preservation (explicit restrictions and negative rules survive)
4. Technical-detail preservation (frameworks, databases, libraries, and tools kept)
5. Output-format preservation (target schema, structures, tables, and formats kept)
6. Downstream usefulness (actionable clarity vs over-compression / loss of utility)

Governing principle:
"Minimize tokens only when doing so does not reduce the prompt's ability to produce the intended result."
Token reduction alone does NOT equal successful optimization.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest
import re

from app.services.prompt_enhancer import PromptEnhancer
from app.services.optimization_intelligence import OptimizationOutcome, OptimizationValue
from app.services.token_utils import compare as compare_tokens


# =========================================================================
# D3 Quality Benchmark Data Structures & Evaluation Engine
# =========================================================================

@dataclass(frozen=True)
class QualityBenchmarkResult:
    """Deterministic quality benchmark assessment across the 6 dimensions."""
    intent_preserved: bool
    requirements_preserved: bool
    constraints_preserved: bool
    technical_details_preserved: bool
    output_format_preserved: bool
    downstream_usefulness_preserved: bool
    quality_preserved: bool


def _text_contains_all(text: str, terms: List[str]) -> bool:
    """Check if all search terms appear in the lowercased text."""
    lower = text.lower()
    for term in terms:
        if term.lower() not in lower:
            return False
    return True


def _text_contains_any(text: str, terms: List[str]) -> bool:
    """Check if any of the search terms appear in the lowercased text."""
    lower = text.lower()
    for term in terms:
        if term.lower() in lower:
            return True
    return False


def evaluate_quality_benchmark(case: Dict[str, Any]) -> QualityBenchmarkResult:
    """
    Deterministic evaluation of prompt candidate against canonical quality dimensions.
    No LLMs used. Pure string/rule-based benchmark verification.
    """
    candidate = case.get("candidate", "")
    candidate_lower = candidate.lower()

    # 1. Intent preservation: candidate must not include forbidden task shifts
    # and must maintain intent keywords
    forbidden = case.get("forbidden_changes", [])
    has_forbidden = _text_contains_any(candidate, forbidden) if forbidden else False
    intent_terms = case.get("intent_terms", [])
    intent_ok = (not has_forbidden) and (_text_contains_all(candidate, intent_terms) if intent_terms else True)

    # 2. Requirements preservation
    req_terms = case.get("required_terms", [])
    reqs_ok = _text_contains_all(candidate, req_terms) if req_terms else True

    # 3. Constraints / negations preservation
    constraints = case.get("required_constraints", [])
    constraints_ok = _text_contains_all(candidate, constraints) if constraints else True

    # 4. Technical details preservation
    tech_terms = case.get("required_technical_terms", [])
    tech_ok = _text_contains_all(candidate, tech_terms) if tech_terms else True

    # 5. Output format preservation
    format_markers = case.get("required_format_markers", [])
    format_ok = _text_contains_all(candidate, format_markers) if format_markers else True

    # 6. Downstream usefulness: must not be an empty or degenerate stub
    usefulness_ok = case.get("usefulness_intact", True) and len(candidate.strip().split()) >= 3

    quality_ok = (
        intent_ok
        and reqs_ok
        and constraints_ok
        and tech_ok
        and format_ok
        and usefulness_ok
    )

    return QualityBenchmarkResult(
        intent_preserved=intent_ok,
        requirements_preserved=reqs_ok,
        constraints_preserved=constraints_ok,
        technical_details_preserved=tech_ok,
        output_format_preserved=format_ok,
        downstream_usefulness_preserved=usefulness_ok,
        quality_preserved=quality_ok,
    )


def _build_mock_response(content: str, total_tokens: int = 50):
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.total_tokens = total_tokens

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    return mock_resp


# =========================================================================
# Canonical D3 Quality Benchmark Cases
# =========================================================================

BENCHMARK_CASES: List[Dict[str, Any]] = [
    # ── Category 1: Intent Preservation ──────────────────────────────────────
    {
        "name": "intent_valid_compression",
        "category": "intent_preservation",
        "mode": "concise",
        "original": "Could you please be so kind as to convert this CSV file into JSON format for me?",
        "candidate": "Convert this CSV file to JSON format.",
        "intent_terms": ["convert", "csv", "json"],
        "required_terms": ["csv", "json"],
        "required_constraints": [],
        "required_technical_terms": ["csv", "json"],
        "required_format_markers": ["json"],
        "forbidden_changes": ["xml", "yaml", "delete", "summarize"],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "intent_adversarial_task_shift",
        "category": "intent_preservation",
        "mode": "concise",
        "original": "Convert this CSV file into JSON format.",
        "candidate": "Parse this XML data and convert it to YAML.",
        "intent_terms": ["convert", "csv", "json"],
        "required_terms": ["csv", "json"],
        "required_constraints": [],
        "required_technical_terms": ["csv", "json"],
        "required_format_markers": ["json"],
        "forbidden_changes": ["xml", "yaml"],
        "expected_quality_preserved": False,
        "expected_decision": "reverted",
    },

    # ── Category 2: Requirement Preservation ─────────────────────────────────
    {
        "name": "requirements_valid_expansion",
        "category": "requirement_preservation",
        "mode": "detailed",
        "original": "Build a FastAPI API that accepts CSV uploads, removes duplicate emails, stores cleaned records in PostgreSQL, and returns the inserted count.",
        "candidate": """Create a FastAPI endpoint with the following requirements:
1. Accept CSV file uploads.
2. Remove duplicate emails.
3. Store cleaned records into PostgreSQL.
4. Return the total count of inserted records.""",
        "intent_terms": ["fastapi", "csv", "postgresql"],
        "required_terms": ["fastapi", "csv", "duplicate", "postgresql", "count"],
        "required_constraints": [],
        "required_technical_terms": ["fastapi", "postgresql", "csv"],
        "required_format_markers": ["1.", "2.", "3.", "4."],
        "forbidden_changes": [],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "requirements_adversarial_dropped_deliverable",
        "category": "requirement_preservation",
        "mode": "detailed",
        "original": "Build a FastAPI API that accepts CSV uploads, removes duplicate emails, stores cleaned records in PostgreSQL, and returns the inserted count.",
        "candidate": "Build a simple script to upload CSV files to PostgreSQL.",
        "intent_terms": ["fastapi", "csv", "postgresql"],
        "required_terms": ["fastapi", "duplicate", "count"],
        "required_constraints": [],
        "required_technical_terms": ["fastapi", "postgresql"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": False,
        "expected_decision": "reverted",
    },

    # ── Category 3: Constraint / Negation Preservation ────────────────────────
    {
        "name": "constraints_valid_refinement",
        "category": "constraint_preservation",
        "mode": "code",
        "original": "Write a user auth service in Node.js. Do not use MongoDB. Must use PostgreSQL. Do not add a frontend.",
        "candidate": """Develop a user authentication service in Node.js adhering to these rules:
- Database: PostgreSQL (do not use MongoDB)
- Backend only: do not add a frontend""",
        "intent_terms": ["user auth", "node.js", "postgresql"],
        "required_terms": ["auth", "node.js"],
        "required_constraints": ["do not use mongodb", "do not add a frontend"],
        "required_technical_terms": ["node.js", "postgresql", "mongodb"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "constraints_adversarial_dropped_negation",
        "category": "constraint_preservation",
        "mode": "code",
        "original": "Write a user auth service in Node.js. Do not use MongoDB. Must use PostgreSQL. Do not add a frontend.",
        "candidate": "Create a user authentication service in Node.js with PostgreSQL and MongoDB.",
        "intent_terms": ["user auth", "node.js"],
        "required_terms": ["auth", "node.js"],
        "required_constraints": ["do not use mongodb", "do not add a frontend"],
        "required_technical_terms": ["node.js", "postgresql"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": False,
        "expected_decision": "reverted",
    },

    # ── Category 4: Technical-Detail Preservation ─────────────────────────────
    {
        "name": "technical_valid_restructure",
        "category": "technical_preservation",
        "mode": "code",
        "original": "Set up a background task worker using Celery with Redis broker, FastAPI integration, and SQLAlchemy async sessions.",
        "candidate": """Configure background task processing:
- Framework: FastAPI with Celery worker
- Broker: Redis
- ORM: SQLAlchemy with async sessions""",
        "intent_terms": ["celery", "redis", "fastapi", "sqlalchemy"],
        "required_terms": ["celery", "redis", "fastapi", "sqlalchemy"],
        "required_constraints": [],
        "required_technical_terms": ["celery", "redis", "fastapi", "sqlalchemy"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "technical_adversarial_generic_replacement",
        "category": "technical_preservation",
        "mode": "code",
        "original": "Set up a background task worker using Celery with Redis broker, FastAPI integration, and SQLAlchemy async sessions.",
        "candidate": "Set up background tasks with a queue and database in Python.",
        "intent_terms": ["celery", "fastapi"],
        "required_terms": ["celery", "redis", "sqlalchemy"],
        "required_constraints": [],
        "required_technical_terms": ["celery", "redis", "fastapi", "sqlalchemy"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": False,
        "expected_decision": "reverted",
    },

    # ── Category 5: Output-Format Preservation ────────────────────────────────
    {
        "name": "output_format_valid_compression",
        "category": "output_format_preservation",
        "mode": "concise",
        "original": "Please provide the response strictly as a JSON object containing keys 'status', 'code', and 'data'.",
        "candidate": "Output strictly as a JSON object with keys: 'status', 'code', 'data'.",
        "intent_terms": ["json", "status", "code", "data"],
        "required_terms": ["json", "status", "code", "data"],
        "required_constraints": [],
        "required_technical_terms": ["json"],
        "required_format_markers": ["json"],
        "forbidden_changes": [],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "output_format_adversarial_lost_json_schema",
        "category": "output_format_preservation",
        "mode": "concise",
        "original": "Please provide the response strictly as a JSON object containing keys 'status', 'code', and 'data'.",
        "candidate": "Summarize the response in a brief text paragraph.",
        "intent_terms": ["json"],
        "required_terms": ["json", "status", "code", "data"],
        "required_constraints": [],
        "required_technical_terms": ["json"],
        "required_format_markers": ["json"],
        "forbidden_changes": ["paragraph"],
        "expected_quality_preserved": False,
        "expected_decision": "reverted",
    },

    # ── Category 6: Downstream Usefulness ─────────────────────────────────────
    {
        "name": "usefulness_valid_refinement",
        "category": "downstream_usefulness",
        "mode": "detailed",
        "original": "explain how oauth2 authorization code flow with JWT tokens works step by step for mobile apps with PostgreSQL backend",
        "candidate": """Explain OAuth2 Authorization Code Flow with JWT tokens and PostgreSQL backend:
1. Code Challenge Generation
2. Authorization Request
3. Authorization Code Issuance
4. JWT Token Exchange
5. Access and Refresh Token Storage in PostgreSQL""",
        "intent_terms": ["oauth2", "jwt", "postgresql"],
        "required_terms": ["oauth2", "jwt", "postgresql", "authorization code"],
        "required_constraints": [],
        "required_technical_terms": ["jwt", "postgresql"],
        "required_format_markers": ["1.", "2.", "3."],
        "forbidden_changes": [],
        "expected_quality_preserved": True,
        "expected_decision": "enhanced",
    },
    {
        "name": "usefulness_adversarial_overcompressed_vague",
        "category": "downstream_usefulness",
        "mode": "detailed",
        "original": "explain how oauth2 authorization code flow with JWT tokens works step by step for mobile apps with PostgreSQL backend",
        "candidate": "Explain OAuth.",
        "intent_terms": ["oauth2", "jwt", "postgresql"],
        "required_terms": ["oauth2", "jwt", "postgresql", "authorization code", "step by step"],
        "required_constraints": [],
        "required_technical_terms": ["jwt", "postgresql"],
        "required_format_markers": [],
        "forbidden_changes": [],
        "expected_quality_preserved": False,
        "usefulness_intact": False,
        "expected_decision": "reverted",
    },
]


# =========================================================================
# Unit & Benchmark Test Suite for D3
# =========================================================================

class TestOptimizationQualityBenchmark:
    """Benchmark tests validating the 6 quality dimensions and adversarial defense."""

    @pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda c: c["name"])
    def test_benchmark_case_quality_evaluation(self, case: Dict[str, Any]):
        """Verify that D3 quality evaluation matches the expected quality preservation state."""
        res = evaluate_quality_benchmark(case)
        assert res.quality_preserved == case["expected_quality_preserved"], (
            f"Case {case['name']} failed quality assertion. Expected {case['expected_quality_preserved']}, got {res}"
        )

    @pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda c: c["name"])
    def test_pipeline_decision_alignment_with_quality(self, case: Dict[str, Any]):
        """
        Verify that PromptEnhancer pipeline decisions align with D3 quality benchmark:
        - Quality-preserved rewrites are accepted (decision == 'enhanced')
        - Quality-damaged rewrites are rejected (decision == 'reverted')
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(
            case["candidate"], total_tokens=50
        )

        enhancer = PromptEnhancer(client=mock_client)
        result = enhancer.enhance(
            case["original"],
            mode=case["mode"],
            bypass_gate=True,
        )

        assert result["decision"] == case["expected_decision"], (
            f"Pipeline decision mismatch for '{case['name']}': expected '{case['expected_decision']}', "
            f"got '{result['decision']}' (reason: {result.get('reason')})"
        )

    def test_adversarial_compression_loss_defense(self):
        """
        Adversarial test: Explicitly demonstrate that token reduction alone != successful optimization.
        A candidate that reduces tokens by 50% but drops critical negations/technologies MUST be reverted.
        """
        original = "Create a FastAPI backend with PostgreSQL. Do not use MongoDB under any circumstance."
        damaging_short_candidate = "Create an API with MongoDB."

        orig_tokens = compare_tokens(original, original)["original_tokens"]
        cand_tokens = compare_tokens(damaging_short_candidate, damaging_short_candidate)["original_tokens"]
        assert cand_tokens < orig_tokens  # Candidate is indeed shorter

        # Quality benchmark must identify failure
        q_res = evaluate_quality_benchmark({
            "candidate": damaging_short_candidate,
            "required_constraints": ["do not use mongodb"],
            "required_technical_terms": ["fastapi", "postgresql"],
        })
        assert q_res.quality_preserved is False
        assert q_res.constraints_preserved is False
        assert q_res.technical_details_preserved is False

        # Production pipeline must reject/revert it
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(damaging_short_candidate)
        enhancer = PromptEnhancer(client=mock_client)
        result = enhancer.enhance(original, mode="code", bypass_gate=True)

        assert result["decision"] == "reverted"
        assert result["enhanced_prompt"] == original  # Safe original kept
        assert result["optimization_outcome"] == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED.value
