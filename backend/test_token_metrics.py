"""
Unit & Integration Tests for PromptPilot Token Metrics & Reporting

Tests:
- Neutral Token Change metric: token_delta = enhanced_tokens - original_tokens
- Token change percent: ((enhanced_tokens - original_tokens) / original_tokens) * 100
- Display formatting:
    Shorter: "X fewer tokens (-Y%)"
    Same:    "No change (0%)"
    Longer:  "X more tokens (+Y%)"
- Positive and negative cases:
    38 -> 21:  17 fewer tokens (-44.7%)
    15 -> 15:  No change (0%)
    68 -> 133: 65 more tokens (+95.6%)
    84 -> 144: 60 more tokens (+71.4%)
    41 -> 46:  5 more tokens (+12.2%)
- Preservation of internal token accounting (tokens_saved, overhead, net tokens)
"""

import pytest
from app.services.token_utils import (
    format_token_change,
    calculate_token_metrics,
    compare as compare_tokens,
)
from app.services.prompt_evaluator import PromptEvaluator
from app.services.prompt_enhancer import PromptEnhancer


def test_token_change_shorter_38_to_21():
    """Test A: 38 -> 21 (prompt became shorter)"""
    metrics = calculate_token_metrics(original_tokens=38, enhanced_tokens=21)

    assert metrics["token_delta"] == -17
    assert metrics["token_change_percent"] == -44.7
    assert metrics["token_change_display"] == "17 fewer tokens (-44.7%)"
    assert "reduction" not in metrics["token_change_display"].lower()

    # Legacy accounting preservation
    assert metrics["tokens_saved"] == 17
    assert metrics["percent_saved"] == 44.7


def test_token_change_same_13_to_13():
    """Test B: 13 -> 13 (no change)"""
    metrics = calculate_token_metrics(original_tokens=13, enhanced_tokens=13)

    assert metrics["token_delta"] == 0
    assert metrics["token_change_percent"] == 0.0
    assert metrics["token_change_display"] == "No change (0%)"

    # Legacy accounting preservation
    assert metrics["tokens_saved"] == 0
    assert metrics["percent_saved"] == 0.0


def test_token_change_same_15_to_15():
    """Additional same test: 15 -> 15"""
    metrics = calculate_token_metrics(original_tokens=15, enhanced_tokens=15)

    assert metrics["token_delta"] == 0
    assert metrics["token_change_percent"] == 0.0
    assert metrics["token_change_display"] == "No change (0%)"


def test_token_change_expansion_57_to_99():
    """Test C: 57 -> 99 (expansion)"""
    metrics = calculate_token_metrics(original_tokens=57, enhanced_tokens=99)

    assert metrics["token_delta"] == 42
    assert metrics["token_change_percent"] == 73.7
    assert metrics["token_change_display"] == "42 more tokens (+73.7%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_token_change_large_expansion_27_to_62():
    """Test D: 27 -> 62 (large expansion)"""
    metrics = calculate_token_metrics(original_tokens=27, enhanced_tokens=62)

    assert metrics["token_delta"] == 35
    assert metrics["token_change_percent"] == 129.6
    assert metrics["token_change_display"] == "35 more tokens (+129.6%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_token_change_small_expansion_41_to_43():
    """Test E: 41 -> 43 (small expansion)"""
    metrics = calculate_token_metrics(original_tokens=41, enhanced_tokens=43)

    assert metrics["token_delta"] == 2
    assert metrics["token_change_percent"] == 4.9
    assert metrics["token_change_display"] == "2 more tokens (+4.9%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_token_change_longer_68_to_133():
    """Expansion: 68 -> 133"""
    metrics = calculate_token_metrics(original_tokens=68, enhanced_tokens=133)

    assert metrics["token_delta"] == 65
    assert metrics["token_change_percent"] == 95.6
    assert metrics["token_change_display"] == "65 more tokens (+95.6%)"
    assert "reduction" not in metrics["token_change_display"].lower()
    assert "-" not in metrics["token_change_display"]

    # Legacy accounting preservation (negative saved)
    assert metrics["tokens_saved"] == -65
    assert metrics["percent_saved"] == -95.6


def test_token_change_longer_84_to_144():
    """Expansion: 84 -> 144"""
    metrics = calculate_token_metrics(original_tokens=84, enhanced_tokens=144)

    assert metrics["token_delta"] == 60
    assert metrics["token_change_percent"] == 71.4
    assert metrics["token_change_display"] == "60 more tokens (+71.4%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_token_change_expansion_51_to_64():
    """Expansion: 51 -> 64"""
    metrics = calculate_token_metrics(original_tokens=51, enhanced_tokens=64)

    assert metrics["token_delta"] == 13
    assert metrics["token_change_percent"] == 25.5
    assert metrics["token_change_display"] == "13 more tokens (+25.5%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_token_change_longer_41_to_46():
    """Expansion: 41 -> 46"""
    metrics = calculate_token_metrics(original_tokens=41, enhanced_tokens=46)

    assert metrics["token_delta"] == 5
    assert metrics["token_change_percent"] == 12.2
    assert metrics["token_change_display"] == "5 more tokens (+12.2%)"
    assert "reduction" not in metrics["token_change_display"].lower()


def test_evaluator_includes_neutral_metrics():
    """Ensure PromptEvaluator output carries both neutral metrics and legacy accounting."""
    evaluator = PromptEvaluator()
    res = evaluator.evaluate(
        original="Create a simple function to sort a list.",
        enhanced="Create a robust, type-annotated Python function to sort a list in ascending or descending order.",
        enhancement_overhead_tokens=150,
    )

    # Neutral metrics
    assert "token_delta" in res
    assert "token_change_percent" in res
    assert "token_change_display" in res
    assert res["token_delta"] > 0
    assert res["token_change_percent"] > 0
    assert "more tokens" in res["token_change_display"]

    # Preserved accounting
    assert "tokens_saved" in res
    assert "percent_saved" in res
    assert "enhancement_overhead_tokens" in res
    assert res["enhancement_overhead_tokens"] == 150
    assert res["total_overhead_tokens"] == 150
    assert res["tokens_saved"] == -res["token_delta"]
