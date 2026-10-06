"""
Regression Tests for Cross-Phase API Contract Schema Invariant

Verifies that every legitimate PromptEnhancer.enhance() return path contains:
- optimization_outcome
- optimization_value
- optimization_efficiency_score

Specifically tests all 8 neutral early/error return paths:
1. Empty/blank prompt
2. Invalid mode
3. Large prompt / cost guard (>2000 tokens)
4. Missing API key / unavailable LLM client (client is None)
5. Fixed-mode exception
6. Auto-mode classifier failure
7. Auto-mode invalid classifier mode
8. Auto-mode exception
"""

from unittest.mock import MagicMock
import pytest

from app.services.prompt_enhancer import PromptEnhancer
from app.services.prompt_classifier import PromptClassifier


class TestApiContractSchemaInvariant:
    """Verify all early/error return paths contain consistent optimization metadata."""

    def _assert_neutral_optimization_metadata(self, result: dict):
        assert "optimization_outcome" in result
        assert "optimization_value" in result
        assert "optimization_efficiency_score" in result
        assert result["optimization_outcome"] == "ALREADY_EFFICIENT"
        assert result["optimization_value"] == "NO_OPTIMIZATION_NEEDED"
        assert result["optimization_efficiency_score"] == 0.0

    def test_path_1_empty_or_blank_prompt(self):
        """Path 1: Empty or whitespace-only prompt."""
        enhancer = PromptEnhancer(client=MagicMock())

        res_empty = enhancer.enhance("")
        self._assert_neutral_optimization_metadata(res_empty)
        assert res_empty["success"] is False

        res_spaces = enhancer.enhance("   \n\t  ")
        self._assert_neutral_optimization_metadata(res_spaces)
        assert res_spaces["success"] is False

    def test_path_2_invalid_mode(self):
        """Path 2: Invalid enhancement mode."""
        enhancer = PromptEnhancer(client=MagicMock())

        res = enhancer.enhance("Write a python script", mode="invalid_super_mode")
        self._assert_neutral_optimization_metadata(res)
        assert res["success"] is False
        assert res["decision"] == "unchanged"

    def test_path_3_large_prompt_cost_guard(self):
        """Path 3: Oversized prompt exceeding 2000 tokens."""
        enhancer = PromptEnhancer(client=MagicMock())

        # Generate ~2500 tokens
        large_prompt = "word " * 3000
        res = enhancer.enhance(large_prompt, mode="concise")
        self._assert_neutral_optimization_metadata(res)
        assert res["success"] is True
        assert res["decision"] == "unchanged"
        assert res["model"] == "local_gate"

    def test_path_4_missing_api_key_or_client(self):
        """Path 4: Missing API key / client is None (bypassing gate)."""
        enhancer = PromptEnhancer(client=MagicMock())
        enhancer.client = None

        res = enhancer.enhance("Refactor this python script to use async", mode="code", bypass_gate=True)
        self._assert_neutral_optimization_metadata(res)
        assert res["success"] is False
        assert res["decision"] == "error"

    def test_path_5_fixed_mode_exception(self):
        """Path 5: Unhandled exception during fixed-mode enhancement."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Fatal groq socket crash")

        enhancer = PromptEnhancer(client=mock_client)
        res = enhancer.enhance("Refactor this python script", mode="code", bypass_gate=True)

        self._assert_neutral_optimization_metadata(res)
        assert res["success"] is False
        assert res["decision"] == "error"

    def test_path_6_auto_classifier_failure(self):
        """Path 6: Auto-mode classifier returns failure."""
        mock_client = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "mode": None,
            "success": False,
            "reason": "Classifier API timed out",
            "tokens": 0,
        }

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance("Make a website", mode="auto", bypass_gate=True)

        self._assert_neutral_optimization_metadata(res)
        assert res["decision"] == "unchanged"

    def test_path_7_auto_invalid_classifier_mode(self):
        """Path 7: Auto-mode classifier returns unrecognized mode."""
        mock_client = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "mode": "unrecognized_mode_xyz",
            "success": True,
            "reason": None,
            "tokens": 15,
        }

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance("Make a website", mode="auto", bypass_gate=True)

        self._assert_neutral_optimization_metadata(res)
        assert res["decision"] == "unchanged"

    def test_path_8_auto_mode_exception(self):
        """Path 8: Unhandled exception in auto-mode."""
        mock_client = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("Unexpected classifier memory error")

        enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
        res = enhancer.enhance("Make a website", mode="auto", bypass_gate=True)

        self._assert_neutral_optimization_metadata(res)
        assert res["success"] is False
        assert res["decision"] == "error"
