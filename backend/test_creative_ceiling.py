"""
Tests for creative-mode small-prompt growth ceiling (Phase A fix).

Verifies:
    - Creative small prompts use 2.5× ratio instead of 96-token absolute.
    - Non-creative small prompts still use 96-token absolute.
    - Concise small prompts still return original_tokens.
    - Creative prompts above threshold use normal ratios.
    - A 68-token candidate for a 17-token creative prompt is rejected.
    - A candidate within the 2.5× ceiling is accepted.
"""

from unittest.mock import MagicMock
import pytest

from app.services.prompt_enhancer import PromptEnhancer


def _mock_groq(content: str, total_tokens: int = 200):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    if total_tokens is not None:
        mock_usage = MagicMock()
        mock_usage.total_tokens = total_tokens
        mock_response.usage = mock_usage
    else:
        mock_response.usage = None

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# =========================================================================
# 1. Growth ceiling values for creative small prompts
# =========================================================================

class TestCreativeGrowthCeiling:
    """Verify _growth_ceiling returns correct values for creative mode."""

    def test_17_token_creative_ceiling(self):
        """17-token creative prompt → ceiling = int(17 * 2.5) = 42."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 17)
        assert ceiling == 42

    def test_8_token_creative_ceiling(self):
        """8-token creative prompt → ceiling = max(36, int(8 * 2.5)) = 36."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 8)
        assert ceiling == 36

    def test_20_token_creative_ceiling(self):
        """20-token creative prompt (at threshold) → ceiling = int(20 * 2.5) = 50."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 20)
        assert ceiling == 50

    def test_1_token_creative_ceiling(self):
        """1-token creative prompt → ceiling = max(36, int(1 * 2.5)) = 36."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 1)
        assert ceiling == 36

    def test_10_token_creative_ceiling(self):
        """10-token creative prompt → ceiling = max(36, int(10 * 2.5)) = 36."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 10)
        assert ceiling == 36

    def test_14_token_creative_ceiling(self):
        """14-token creative prompt → ceiling = max(36, int(14 * 2.5)) = 36."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 14)
        assert ceiling == 36

    def test_15_token_creative_ceiling(self):
        """15-token creative prompt → ceiling = max(36, int(15 * 2.5)) = 37."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 15)
        assert ceiling == 37

    def test_creative_above_threshold_uses_normal_ratio(self):
        """Creative prompt above threshold → uses 2.0× ratio, not 2.5×."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 25)
        assert ceiling == int(25 * 2.0)  # 50, not 62

    def test_creative_immediately_above_threshold(self):
        """21-token creative prompt immediately above threshold (20) → uses 2.0× ratio (42)."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 21)
        assert ceiling == 42

    def test_creative_immediately_above_threshold_structured(self):
        """21-token creative prompt structured immediately above threshold → uses 2.5× ratio (52)."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 21, structured=True)
        assert ceiling == 52

    def test_creative_above_threshold_structured(self):
        """Creative prompt above threshold with structure → 2.5× structured ratio."""
        ceiling = PromptEnhancer._growth_ceiling("creative", 25, structured=True)
        assert ceiling == int(25 * 2.5)  # 62


# =========================================================================
# 2. Non-creative modes are unaffected
# =========================================================================

class TestNonCreativeUnaffected:
    """Verify non-creative small prompts still use the 96-token absolute ceiling."""

    def test_detailed_small_prompt_uses_absolute_ceiling(self):
        ceiling = PromptEnhancer._growth_ceiling("detailed", 17)
        assert ceiling == 96

    def test_code_small_prompt_uses_absolute_ceiling(self):
        ceiling = PromptEnhancer._growth_ceiling("code", 17)
        assert ceiling == 96

    def test_concise_small_prompt_returns_original(self):
        """Concise mode never grows, regardless of prompt size."""
        ceiling = PromptEnhancer._growth_ceiling("concise", 17)
        assert ceiling == 17


# =========================================================================
# 3. Acceptance policy rejects over-expanded creative prompts
# =========================================================================

class TestCreativeAcceptancePolicy:
    """Verify the full acceptance pipeline with creative ceiling."""

    def test_68_token_creative_candidate_rejected(self):
        """
        A 17-token creative prompt producing a 68-token candidate must
        be rejected: 68 > 42 (ceiling at 2.5×).
        """
        original = (
            "Give me practical AI startup ideas for developers "
            "that are different from common chatbot products."
        )

        # Simulate an over-expanded creative output (the old failure mode)
        over_expanded = (
            "Brainstorm practical AI startup ideas for developers "
            "that move beyond standard chatbot products. For each idea, outline:\n"
            "- The specific developer pain point it solves\n"
            "- The core AI capability or workflow it leverages\n"
            "- A brief description of the product experience\n"
            "Focus on distinct, actionable concepts that offer clear "
            "value to engineering teams or individual developers."
        )

        mock_client = _mock_groq(content=over_expanded, total_tokens=300)
        enhancer = PromptEnhancer(client=mock_client)

        result = enhancer.enhance(original, mode="creative", bypass_gate=True)

        assert result["decision"] == "reverted"
        assert result["enhanced_prompt"] == original
        assert "safety ceiling" in result["reason"] or "likely invented" in result["reason"]

    def test_short_creative_candidate_accepted(self):
        """
        A 17-token creative prompt producing a ~30-token candidate
        should be accepted: 30 < 42 (ceiling at 2.5×).
        """
        original = (
            "Give me practical AI startup ideas for developers "
            "that are different from common chatbot products."
        )

        # A good, minimal creative enhancement
        good_candidate = (
            "Brainstorm practical AI startup ideas for developers "
            "that go beyond standard chatbot products. Focus on "
            "distinct, actionable concepts."
        )

        mock_client = _mock_groq(content=good_candidate, total_tokens=250)
        enhancer = PromptEnhancer(client=mock_client)

        result = enhancer.enhance(original, mode="creative", bypass_gate=True)

        assert result["decision"] == "enhanced"
        assert result["enhanced_prompt"] == good_candidate
