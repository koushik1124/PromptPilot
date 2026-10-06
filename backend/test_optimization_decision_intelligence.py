"""
Unit & Integration Tests for Phase D2: Optimization Decision Intelligence

Tests:
1. Compression classification: HIGH_VALUE, MODERATE, LOW_VALUE with absolute delta guards
2. Expansion classification: JUSTIFIED_EXPANSION vs LOW_VALUE_EXPANSION with structural improvement
3. REFINE strategy wording polish classification by delta band
4. Strategy distinction (STRUCTURE, REFINE, CREATIVE_EXPAND, COMPRESS, NO_CHANGE)
5. Non-accepted decisions (unchanged, reverted, error, bypass outcomes)
6. Edge cases (zero tokens, 1 token, large prompts, boundary values)
7. Integration tests with PromptEnhancer pipeline
"""

from unittest.mock import MagicMock
import pytest

from app.services.optimization_intelligence import (
    OptimizationOutcome,
    OptimizationValue,
    OptimizationValueResult,
    classify_optimization_outcome,
    classify_optimization_value,
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
# 1. Compression Classification Tests
# =========================================================================

class TestCompressionValueClassification:
    """Tests for compression classification rules and thresholds."""

    def test_c1_low_value_compression_minimal(self):
        """100 -> 99 (-1 token, -1%) -> LOW_VALUE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=99,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.LOW_VALUE_COMPRESSION
        assert 0.01 <= res.efficiency_score <= 0.05
        assert res.compression_ratio == 0.01
        assert res.token_delta == -1
        assert res.token_delta_percent == -1.0

    def test_c2_low_value_compression_below_5_percent(self):
        """100 -> 97 (-3 tokens, -3%) -> LOW_VALUE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=97,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.LOW_VALUE_COMPRESSION
        assert res.efficiency_score == 0.05
        assert res.compression_ratio == 0.03

    def test_c3_moderate_compression_10_percent(self):
        """100 -> 90 (-10 tokens, -10%) -> MODERATE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=90,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.MODERATE_COMPRESSION
        assert res.efficiency_score == 0.50
        assert res.compression_ratio == 0.10

    def test_c4_high_value_compression_50_percent(self):
        """100 -> 50 (-50 tokens, -50%) -> HIGH_VALUE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=50,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 1.0
        assert res.compression_ratio == 0.50

    def test_c5_high_value_compression_capped(self):
        """100 -> 10 (-90 tokens, -90%) -> HIGH_VALUE_COMPRESSION (capped at 1.0)."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=10,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 1.0
        assert res.compression_ratio == 0.90

    def test_c6_high_value_compression_20_percent_boundary(self):
        """100 -> 80 (-20 tokens, -20%) -> HIGH_VALUE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=100,
            enhanced_tokens=80,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 0.82
        assert res.compression_ratio == 0.20

    def test_c7_high_value_compression_large_prompt(self):
        """1000 -> 500 (-500 tokens, -50%) -> HIGH_VALUE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=1000,
            enhanced_tokens=500,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 1.0
        assert res.compression_ratio == 0.50

    def test_c8_small_prompt_1_token_delta_guard(self):
        """10 -> 9 (-1 token, -10%) -> LOW_VALUE_COMPRESSION due to abs_delta < 3 guard."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=10,
            enhanced_tokens=9,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.LOW_VALUE_COMPRESSION
        assert res.efficiency_score == 0.10

    def test_c9_small_prompt_3_token_delta_high_value(self):
        """10 -> 7 (-3 tokens, -30%) -> HIGH_VALUE_COMPRESSION (ratio >= 0.20 AND abs_delta >= 3)."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=10,
            enhanced_tokens=7,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 0.88
        assert res.compression_ratio == 0.30

    def test_c10_small_prompt_2_token_delta_guard(self):
        """10 -> 8 (-2 tokens, -20%) -> LOW_VALUE_COMPRESSION (ratio >= 0.20 but abs_delta < 3)."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=10,
            enhanced_tokens=8,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.LOW_VALUE_COMPRESSION
        assert res.efficiency_score == 0.20


# =========================================================================
# 2. Expansion Classification Tests
# =========================================================================

class TestExpansionValueClassification:
    """Tests for expansion classification under STRUCTURE, CREATIVE_EXPAND, etc."""

    def test_e1_justified_expansion_minimal(self):
        """100 -> 101, STRUCTURE, struct_improv=True -> JUSTIFIED_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=101,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.80
        assert res.expansion_ratio == 0.01

    def test_e2_justified_expansion_5_percent(self):
        """100 -> 105, STRUCTURE, struct_improv=True -> JUSTIFIED_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=105,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.78
        assert res.expansion_ratio == 0.05

    def test_e3_justified_expansion_20_percent(self):
        """100 -> 120, STRUCTURE, struct_improv=True -> JUSTIFIED_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=120,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.72
        assert res.expansion_ratio == 0.20

    def test_e4_justified_expansion_50_percent(self):
        """100 -> 150, STRUCTURE, struct_improv=True -> JUSTIFIED_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=150,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.60
        assert res.expansion_ratio == 0.50

    def test_e5_low_value_expansion_no_structure(self):
        """100 -> 130, STRUCTURE, struct_improv=False -> LOW_VALUE_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=130,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=False,
        )
        assert res.value == OptimizationValue.LOW_VALUE_EXPANSION
        assert res.efficiency_score == 0.10
        assert res.expansion_ratio == 0.30

    def test_e6_justified_expansion_ratio_over_1(self):
        """10 -> 26 (ratio 1.6 > 1.0), STRUCTURE, struct_improv=True -> JUSTIFIED_EXPANSION with floor 0.35."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=10,
            enhanced_tokens=26,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="code",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.35
        assert res.expansion_ratio == 1.60

    def test_e7_creative_expansion_inherently_justified(self):
        """10 -> 26, CREATIVE_EXPAND, mode=creative -> JUSTIFIED_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=10,
            enhanced_tokens=26,
            strategy=EnhancementStrategy.CREATIVE_EXPAND,
            mode="creative",
            decision="enhanced",
            structural_improvement=False,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.35

    def test_e8_non_structural_expansion_low_value(self):
        """100 -> 130, COMPRESS strategy (unexpected expansion) -> LOW_VALUE_EXPANSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=100,
            enhanced_tokens=130,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
            structural_improvement=False,
        )
        assert res.value == OptimizationValue.LOW_VALUE_EXPANSION
        assert res.efficiency_score == 0.10


# =========================================================================
# 3. REFINE Wording Refinement Tests
# =========================================================================

class TestRefineValueClassification:
    """Tests for REFINE strategy classification across delta bands."""

    def test_r1_refine_zero_delta(self):
        """24 -> 24 (delta=0) -> WORDING_REFINEMENT with score 0.50."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.WORDING_REFINEMENT,
            original_tokens=24,
            enhanced_tokens=24,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.WORDING_REFINEMENT
        assert res.efficiency_score == 0.50
        assert res.token_delta == 0
        assert res.expansion_ratio is None

    def test_r2_refine_plus_1_token(self):
        """24 -> 25 (delta=+1) -> WORDING_REFINEMENT with score 0.45."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.WORDING_REFINEMENT,
            original_tokens=24,
            enhanced_tokens=25,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.WORDING_REFINEMENT
        assert res.efficiency_score == 0.45
        assert res.token_delta == 1

    def test_r3_refine_plus_4_tokens(self):
        """24 -> 28 (delta=+4) -> WORDING_REFINEMENT with score 0.45."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.WORDING_REFINEMENT,
            original_tokens=24,
            enhanced_tokens=28,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.WORDING_REFINEMENT
        assert res.efficiency_score == 0.45
        assert res.token_delta == 4

    def test_r4_refine_plus_7_tokens(self):
        """24 -> 31 (delta=+7) -> WORDING_REFINEMENT with score 0.35."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.WORDING_REFINEMENT,
            original_tokens=24,
            enhanced_tokens=31,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.WORDING_REFINEMENT
        assert res.efficiency_score == 0.35
        assert res.token_delta == 7

    def test_r5_refine_plus_8_tokens(self):
        """24 -> 32 (delta=+8) -> WORDING_REFINEMENT with score 0.35."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.WORDING_REFINEMENT,
            original_tokens=24,
            enhanced_tokens=32,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.WORDING_REFINEMENT
        assert res.efficiency_score == 0.35
        assert res.token_delta == 8

    def test_r6_refine_compressed(self):
        """50 -> 45 (delta=-5) under REFINE -> MODERATE_COMPRESSION."""
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=50,
            enhanced_tokens=45,
            strategy=EnhancementStrategy.REFINE,
            mode="detailed",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.MODERATE_COMPRESSION
        assert res.efficiency_score == 0.50
        assert res.compression_ratio == 0.10


# =========================================================================
# 4. Non-Accepted Decisions & Bypass States
# =========================================================================

class TestNonAcceptedDecisions:
    """Tests that non-accepted or bypass states return NO_OPTIMIZATION_NEEDED."""

    def test_d1_unchanged_decision(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.ALREADY_EFFICIENT,
            original_tokens=30,
            enhanced_tokens=30,
            decision="unchanged",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_d2_reverted_decision(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED,
            original_tokens=30,
            enhanced_tokens=30,
            decision="reverted",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_d3_error_decision(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.ALREADY_EFFICIENT,
            original_tokens=30,
            enhanced_tokens=30,
            decision="error",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_d4_already_efficient_outcome(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.ALREADY_EFFICIENT,
            original_tokens=30,
            enhanced_tokens=30,
            decision="enhanced",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_d5_preserved_ambiguous_outcome(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.PRESERVED_AMBIGUOUS,
            original_tokens=10,
            enhanced_tokens=10,
            decision="enhanced",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_d6_unnecessary_expansion_prevented_outcome(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.UNNECESSARY_EXPANSION_PREVENTED,
            original_tokens=20,
            enhanced_tokens=20,
            decision="enhanced",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0


# =========================================================================
# 5. Edge Cases & Robustness
# =========================================================================

class TestEdgeCasesAndRobustness:
    """Tests edge cases: 0 tokens, 1 token, large prompts, zero delta non-refine."""

    def test_z1_zero_tokens_safe(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.ALREADY_EFFICIENT,
            original_tokens=0,
            enhanced_tokens=0,
            decision="unchanged",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0
        assert res.token_delta == 0

    def test_z2_zero_original_tokens_safe_division(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.USEFUL_EXPANSION,
            original_tokens=0,
            enhanced_tokens=5,
            strategy=EnhancementStrategy.STRUCTURE,
            mode="detailed",
            decision="enhanced",
            structural_improvement=True,
        )
        assert res.value == OptimizationValue.JUSTIFIED_EXPANSION
        assert res.efficiency_score == 0.35

    def test_z3_single_token_no_change(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.ALREADY_EFFICIENT,
            original_tokens=1,
            enhanced_tokens=1,
            decision="unchanged",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0

    def test_z4_very_large_prompt_compression(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=5000,
            enhanced_tokens=2500,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.HIGH_VALUE_COMPRESSION
        assert res.efficiency_score == 1.0

    def test_z5_zero_delta_non_refine(self):
        res = classify_optimization_value(
            outcome=OptimizationOutcome.COMPRESSED,
            original_tokens=50,
            enhanced_tokens=50,
            strategy=EnhancementStrategy.COMPRESS,
            mode="concise",
            decision="enhanced",
        )
        assert res.value == OptimizationValue.NO_OPTIMIZATION_NEEDED
        assert res.efficiency_score == 0.0


# =========================================================================
# 6. Pipeline Integration Tests via PromptEnhancer
# =========================================================================

class TestEnhancerPipelineD2Integration:
    """Verify PromptEnhancer.enhance() populates optimization_value and efficiency score."""

    def test_i1_concise_compression_pipeline(self):
        """Conversational filler compressed in concise mode returns HIGH_VALUE_COMPRESSION."""
        original = (
            "Hey could you please be so kind and go ahead and write me a quick python script "
            "that takes a csv file and removes duplicate rows from it thanks so much"
        )
        enhanced = "Write a Python script to remove duplicate rows from a CSV file."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(enhanced, 60)

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(original, mode="concise", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["optimization_outcome"] == OptimizationOutcome.COMPRESSED.value
        assert res["optimization_value"] == OptimizationValue.HIGH_VALUE_COMPRESSION.value
        assert res["optimization_efficiency_score"] >= 0.70

    def test_i2_detailed_structured_expansion_pipeline(self):
        """Messy prompt restructured with bullets returns JUSTIFIED_EXPANSION."""
        original = "need a rest api for user auth login register jwt refresh tokens postgresql"
        enhanced = (
            "Create a REST API with authentication:\n"
            "1. User Registration and Login\n"
            "2. JWT Access and Refresh Tokens\n"
            "3. PostgreSQL database integration"
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_response(enhanced, 70)

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(original, mode="detailed", bypass_gate=True)

        assert res["decision"] == "enhanced"
        assert res["optimization_outcome"] == OptimizationOutcome.USEFUL_EXPANSION.value
        assert res["optimization_value"] == OptimizationValue.JUSTIFIED_EXPANSION.value
        assert res["optimization_efficiency_score"] >= 0.40

    def test_i3_gate_bypass_pipeline(self):
        """Direct prompt bypassed by pre-gate returns NO_OPTIMIZATION_NEEDED."""
        prompt = "Write a python function to add two numbers."
        mock_client = MagicMock()

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance(prompt, mode="code")

        assert res["decision"] == "unchanged"
        assert res["optimization_outcome"] == OptimizationOutcome.ALREADY_EFFICIENT.value
        assert res["optimization_value"] == OptimizationValue.NO_OPTIMIZATION_NEEDED.value
        assert res["optimization_efficiency_score"] == 0.0
