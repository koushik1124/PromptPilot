from unittest.mock import MagicMock
import pytest
from app.services.prompt_enhancer import PromptEnhancer
from app.services.strategy_router import EnhancementStrategy
from app.services.strategy_rules import get_strategy_rule


def _build_mock_response(content: str = "Enhanced prompt", total_tokens: int = 42):
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.total_tokens = total_tokens

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    return mock_resp


def test_no_change_bypasses_llm_in_auto():
    mock_client = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "code",
        "tokens": 15,
        "success": True,
        "reason": "Structured code task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = """Build a FastAPI backend.

Requirements:
- Use PostgreSQL
- Use SQLAlchemy

Constraints:
- Do not use MongoDB
"""

    # bypass_gate=True to verify strategy router NO_CHANGE bypass
    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.NO_CHANGE.value
    assert result["decision"] == "unchanged"
    assert result["model"] == "strategy_router"
    assert result["enhanced_prompt"] == prompt.strip()
    assert result["enhancement_overhead_tokens"] == 0
    mock_client.chat.completions.create.assert_not_called()


def test_strategy_rule_injected_in_system_prompt_for_compress():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        "Summarize this text in 3 bullet points."
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "concise",
        "tokens": 12,
        "success": True,
        "reason": "Concise task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = "Can you please be so kind and summarize this text for me"

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.COMPRESS.value
    assert mock_client.chat.completions.create.call_count == 1

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_content = messages[0]["content"]

    expected_rule = get_strategy_rule(EnhancementStrategy.COMPRESS.value)
    assert expected_rule in system_content
    assert "Strategy instruction:" in system_content


def test_strategy_rule_injected_in_system_prompt_for_structure():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        "Write a Python script to parse CSV files and remove duplicates."
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "code",
        "tokens": 14,
        "success": True,
        "reason": "Code task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = "write a python script to parse csv and remove duplicates and save to file"

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.STRUCTURE.value

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]

    expected_rule = get_strategy_rule(EnhancementStrategy.STRUCTURE.value)
    assert expected_rule in system_content


def test_strategy_rule_injected_in_system_prompt_for_refine():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        "Build a FastAPI backend.\n\nRequirements:\n- Use PostgreSQL and SQLAlchemy."
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "code",
        "tokens": 14,
        "success": True,
        "reason": "Code task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = "Build a FastAPI backend.\n\nRequirements:\nUse PostgreSQL and SQLAlchemy."

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.REFINE.value

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]

    expected_rule = get_strategy_rule(EnhancementStrategy.REFINE.value)
    assert expected_rule in system_content


def test_strategy_rule_injected_in_system_prompt_for_creative_expand():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        "Write a short story about a solitary clockmaker in a misty mountain village."
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "creative",
        "tokens": 10,
        "success": True,
        "reason": "Creative task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = "write a short story about a clockmaker"

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.CREATIVE_EXPAND.value

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]

    expected_rule = get_strategy_rule(EnhancementStrategy.CREATIVE_EXPAND.value)
    assert expected_rule in system_content


def test_fixed_mode_without_strategy_preserves_legacy_behavior():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        "Write a Python function to compute fibonacci numbers."
    )

    enhancer = PromptEnhancer(client=mock_client)
    prompt = "write a python function to compute fibonacci numbers with memoization"

    result = enhancer.enhance(prompt, mode="code", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] is None
    assert result["mode"] == "code"

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "Strategy instruction:" not in system_content


def test_refine_strategy_for_multi_sentence_fastapi():
    """Integration test for the FastAPI/PostgreSQL/JWT benchmark prompt.

    As multi-sentence clear prose, it routes to REFINE to avoid
    heavy structural reorganization into categories.
    """

    compact_response = (
        "Build a FastAPI backend for a task management application using:\n"
        "- PostgreSQL\n"
        "- SQLAlchemy\n"
        "- Pydantic\n"
        "- JWT authentication\n"
        "\n"
        "Constraint: Do not use MongoDB."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        compact_response, total_tokens=50,
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "code",
        "tokens": 12,
        "success": True,
        "reason": "Code task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = (
        "Build a FastAPI backend for a task management application "
        "using PostgreSQL, SQLAlchemy, Pydantic, and JWT authentication. "
        "Do not use MongoDB."
    )

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.REFINE.value

    # Verify the system prompt uses the REFINE strategy rule
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    strategy_rule = get_strategy_rule(EnhancementStrategy.REFINE.value)
    assert strategy_rule in system_content

    assert result["decision"] == "enhanced"

    enhanced = result["enhanced_prompt"].lower()
    for term in ["fastapi", "postgresql", "sqlalchemy", "pydantic", "jwt"]:
        assert term in enhanced, f"Technology '{term}' missing from enhanced prompt"
    assert "mongodb" in enhanced, "MongoDB exclusion missing from enhanced prompt"
    assert "do not" in enhanced or "avoid" in enhanced, (
        "Negation cue for MongoDB exclusion missing"
    )


def test_refine_strategy_for_react_dashboard_benchmark():
    """Integration test for the React dashboard benchmark prompt.

    Multi-sentence prompt should route to REFINE for minimal polish.
    """

    polished_response = (
        "Create a React dashboard for monitoring cloud servers that displays "
        "real-time charts for CPU usage, memory consumption, and network traffic."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        polished_response, total_tokens=40,
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "mode": "code",
        "tokens": 12,
        "success": True,
        "reason": "Code task",
    }

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    prompt = (
        "Create a React dashboard for monitoring cloud servers. "
        "It should display CPU usage, memory consumption, and network traffic in real-time charts."
    )

    result = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert result["success"] is True
    assert result["strategy"] == EnhancementStrategy.REFINE.value

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    strategy_rule = get_strategy_rule(EnhancementStrategy.REFINE.value)
    assert strategy_rule in system_content

    assert result["decision"] == "enhanced"

