"""
Unit & Integration Tests for Phase D1: Token Intelligence & Outcome Classification

Tests:
1. ALREADY_EFFICIENT: Direct task or already-structured pre-gate / router bypass
2. PRESERVED_AMBIGUOUS: Ambiguous prompt pre-gate bypass ("fix this", "write code", "make it better")
3. COMPRESSED: Accepted shorter candidate (token_delta < 0)
4. WORDING_REFINEMENT: Accepted REFINE wording polish (+7, +8 tokens within tolerance)
5. UNNECESSARY_EXPANSION_PREVENTED: Rejected candidates (+9 token REFINE, ceiling breaches, structure missing)
6. USEFUL_EXPANSION: Accepted STRUCTURE and CREATIVE_EXPAND expansions
7. Rejected shorter candidate: Shorter candidate that was rejected (e.g. dropped negations) must NOT be COMPRESSED
8. Pipeline end-to-end integration via PromptEnhancer
"""

from unittest.mock import MagicMock
import pytest

from app.services.optimization_intelligence import (
    OptimizationOutcome,
    classify_optimization_outcome,
)
from app.services.optimization_gate import OptimizationGate
from app.services.prompt_enhancer import PromptEnhancer
from app.services.strategy_router import EnhancementStrategy


def _build_mock_response(content: str = "Enhanced prompt", total_tokens: int = 50):
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
# 1. Direct Classifier Unit Tests
# =========================================================================

class TestClassifyOptimizationOutcomeUnit:
    """Deterministic unit tests for classify_optimization_outcome()."""

    def test_already_efficient_gate_bypass(self):
        """Pre-gate direct task / structured bypass → ALREADY_EFFICIENT."""
        gate_res = {"should_optimize": False, "is_ambiguous": False}
        outcome = classify_optimization_outcome(
            decision="unchanged",
            original_tokens=20,
            enhanced_tokens=20,
            gate_result=gate_res,
        )
        assert outcome == OptimizationOutcome.ALREADY_EFFICIENT

    def test_preserved_ambiguous_gate_bypass(self):
        """Pre-gate ambiguous bypass → PRESERVED_AMBIGUOUS."""
        gate_res = {"should_optimize": False, "is_ambiguous": True}
        outcome = classify_optimization_outcome(
            decision="unchanged",
            original_tokens=5,
            enhanced_tokens=5,
            gate_result=gate_res,
        )
        assert outcome == OptimizationOutcome.PRESERVED_AMBIGUOUS

    def test_compressed_accepted(self):
        """Accepted candidate with token reduction → COMPRESSED."""
        outcome = classify_optimization_outcome(
            decision="enhanced",
            original_tokens=50,
            enhanced_tokens=30,
            strategy=EnhancementStrategy.COMPRESS,
        )
        assert outcome == OptimizationOutcome.COMPRESSED

    def test_wording_refinement_accepted_plus_7(self):
        """Accepted REFINE candidate with +7 token polish → WORDING_REFINEMENT."""
        outcome = classify_optimization_outcome(
            decision="enhanced",
            original_tokens=24,
            enhanced_tokens=31,  # +7 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert outcome == OptimizationOutcome.WORDING_REFINEMENT

    def test_wording_refinement_accepted_plus_8(self):
        """Accepted REFINE candidate with +8 token polish → WORDING_REFINEMENT."""
        outcome = classify_optimization_outcome(
            decision="enhanced",
            original_tokens=24,
            enhanced_tokens=32,  # +8 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert outcome == OptimizationOutcome.WORDING_REFINEMENT

    def test_unnecessary_expansion_prevented_reverted_refine_plus_9(self):
        """Rejected candidate with +9 token growth → UNNECESSARY_EXPANSION_PREVENTED."""
        outcome = classify_optimization_outcome(
            decision="reverted",
            original_tokens=24,
            enhanced_tokens=24,
            candidate_tokens=33,  # +9 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert outcome == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED

    def test_useful_expansion_structure(self):
        """Accepted candidate with STRUCTURE growth → USEFUL_EXPANSION."""
        outcome = classify_optimization_outcome(
            decision="enhanced",
            original_tokens=30,
            enhanced_tokens=55,
            strategy=EnhancementStrategy.STRUCTURE,
        )
        assert outcome == OptimizationOutcome.USEFUL_EXPANSION

    def test_useful_expansion_creative(self):
        """Accepted candidate with CREATIVE_EXPAND growth → USEFUL_EXPANSION."""
        outcome = classify_optimization_outcome(
            decision="enhanced",
            original_tokens=10,
            enhanced_tokens=26,
            strategy=EnhancementStrategy.CREATIVE_EXPAND,
            mode="creative",
        )
        assert outcome == OptimizationOutcome.USEFUL_EXPANSION

    def test_rejected_shorter_candidate_not_compressed(self):
        """
        Candidate is shorter (15 tokens vs 20) but dropped critical negations and was reverted.
        Must NOT be classified as COMPRESSED.
        """
        outcome = classify_optimization_outcome(
            decision="reverted",
            original_tokens=20,
            enhanced_tokens=20,
            candidate_tokens=15,
            strategy=EnhancementStrategy.COMPRESS,
        )
        assert outcome != OptimizationOutcome.COMPRESSED
        assert outcome == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED

    def test_rejected_growth_unnecessary_expansion_prevented(self):
        """Candidate grew but was reverted → UNNECESSARY_EXPANSION_PREVENTED."""
        outcome = classify_optimization_outcome(
            decision="reverted",
            original_tokens=20,
            enhanced_tokens=20,
            candidate_tokens=60,
            strategy=EnhancementStrategy.STRUCTURE,
        )
        assert outcome == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED


# =========================================================================
# 2. Pipeline End-to-End Integration Tests
# =========================================================================

class TestPipelineOptimizationIntelligence:
    """Verify PromptEnhancer.enhance() returns the correct optimization_outcome field."""

    def test_pipeline_direct_task_already_efficient(self):
        """Direct short task bypassed by gate → ALREADY_EFFICIENT."""
        prompt = "Sort a list of dictionaries by key 'timestamp' in Python."
        mock_client = MagicMock()
        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt, mode="code")

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.ALREADY_EFFICIENT.value
        mock_client.chat.completions.create.assert_not_called()

    def test_pipeline_already_structured_already_efficient(self):
        """Fully structured spec bypassed → ALREADY_EFFICIENT."""
        prompt = (
            "Design a Payment Gateway in Go.\n\n"
            "Requirements:\n"
            "- Idempotency validation\n"
            "- Stripe adapter"
        )
        mock_client = MagicMock()
        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt, mode="auto")

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.ALREADY_EFFICIENT.value
        mock_client.chat.completions.create.assert_not_called()

    def test_pipeline_ambiguous_fix_this(self):
        """Ambiguous prompt 'fix this' → PRESERVED_AMBIGUOUS."""
        prompt = "fix this"
        mock_client = MagicMock()
        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt)

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.PRESERVED_AMBIGUOUS.value
        mock_client.chat.completions.create.assert_not_called()

    def test_pipeline_ambiguous_write_code(self):
        """Ambiguous prompt 'write code' → PRESERVED_AMBIGUOUS."""
        prompt = "write code"
        mock_client = MagicMock()
        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt)

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.PRESERVED_AMBIGUOUS.value

    def test_pipeline_ambiguous_make_it_better(self):
        """Ambiguous prompt 'make it better' → PRESERVED_AMBIGUOUS."""
        prompt = "make it better"
        mock_client = MagicMock()
        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt)

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.PRESERVED_AMBIGUOUS.value

    def test_pipeline_compressed_conversational(self):
        """Polite conversational fluff compressed → COMPRESSED."""
        prompt = (
            "Hello there! I was wondering if you could please kindly help me by writing "
            "a Python script to convert Markdown text to HTML format? Thank you so much!"
        )
        compressed = "Write a Python script to convert Markdown text to HTML format."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(compressed, total_tokens=25)

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt, mode="concise", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["optimization_outcome"] == OptimizationOutcome.COMPRESSED.value
        assert res["token_delta"] < 0

    def test_pipeline_refine_wording_refinement(self):
        """Case 10 GCD prompt with REFINE wording refinement → WORDING_REFINEMENT."""
        prompt = (
            "Write a Python function that calculate the GCD of two numbers using Euclidean algorithm. "
            "Include docstring and type hints."
        )
        refined = (
            "Write a Python function that calculates the greatest common divisor (GCD) of two numbers "
            "using the Euclidean algorithm. Include a docstring and type hints."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(refined, total_tokens=40)
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {"mode": "code", "tokens": 10, "success": True}

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["strategy"] == EnhancementStrategy.REFINE.value
        assert res["optimization_outcome"] == OptimizationOutcome.WORDING_REFINEMENT.value

    def test_pipeline_structure_useful_expansion(self):
        """Unstructured run-on sentence structured into steps → USEFUL_EXPANSION."""
        prompt = (
            "Write a python script to download sales data from s3 bucket parse "
            "csv rows aggregate monthly totals by region and output summary to parquet file."
        )
        structured = (
            "Write a Python script that performs the following steps:\n"
            "1. Downloads sales data from an S3 bucket.\n"
            "2. Parses CSV rows.\n"
            "3. Aggregates monthly totals by region.\n"
            "4. Exports summary dataset to Parquet."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(structured, total_tokens=65)
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {"mode": "code", "tokens": 10, "success": True}

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["strategy"] == EnhancementStrategy.STRUCTURE.value
        assert res["optimization_outcome"] == OptimizationOutcome.USEFUL_EXPANSION.value

    def test_pipeline_creative_useful_expansion(self):
        """Case 15 short story creative expansion → USEFUL_EXPANSION."""
        prompt = "Write a short story about an astronaut stranded on Mars."
        expanded = (
            "Write a short science fiction story about an astronaut stranded on Mars.\n"
            "Develop the protagonist's survival struggle, psychological state, and environmental obstacles, "
            "leading toward a compelling resolution."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(expanded, total_tokens=45)
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {"mode": "creative", "tokens": 8, "success": True}

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["strategy"] == EnhancementStrategy.CREATIVE_EXPAND.value
        assert res["optimization_outcome"] == OptimizationOutcome.USEFUL_EXPANSION.value

    def test_pipeline_hallucinated_growth_unnecessary_expansion_prevented(self):
        """Massive hallucinated output reverted → UNNECESSARY_EXPANSION_PREVENTED."""
        original = "Write a Python function to sort a dictionary by its values."
        hallucinated = (
            "Create an enterprise microservice architecture with Docker, Kubernetes, AWS Lambda, "
            "Terraform CI/CD pipelines, OAuth2 authentication, Redis caching, Prometheus metrics, "
            "GraphQL gateway, Elasticsearch indexing, and write a Python function: "
            + " ".join(["extra invented requirement"] * 60)
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(hallucinated, total_tokens=450)

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(original, mode="code", bypass_gate=True)

        assert res["decision"] == "reverted"
        assert res["optimization_outcome"] == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED.value
        assert res["enhanced_prompt"] == original

    def test_pipeline_dropped_negation_shorter_not_compressed(self):
        """
        Shorter candidate that dropped explicit negation is reverted → UNNECESSARY_EXPANSION_PREVENTED.
        Must NOT be COMPRESSED.
        """
        original = "Build a REST API in Go. Do not use global database variables."
        # Shorter candidate that dropped the negation
        dropped_negation = "Build a REST API in Go."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(dropped_negation, total_tokens=20)

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(original, mode="code", bypass_gate=True)

        assert res["decision"] == "reverted"
        assert res["optimization_outcome"] != OptimizationOutcome.COMPRESSED.value
        assert res["optimization_outcome"] == OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED.value
