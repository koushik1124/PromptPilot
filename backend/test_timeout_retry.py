"""
Unit and Integration Tests for Timeout and Retry Hardening (E1-1)
"""

from unittest.mock import MagicMock
import pytest

from app.services.prompt_classifier import PromptClassifier
from app.services.prompt_enhancer import PromptEnhancer


def _mock_success_response(content: str = "Enhanced prompt", tokens: int = 150):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.finish_reason = "stop"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_usage = MagicMock()
    mock_usage.total_tokens = tokens
    mock_response.usage = mock_usage
    return mock_response


def test_enhancer_timeout_and_retry_constants():
    assert PromptEnhancer._REQUEST_TIMEOUT_SECONDS == 15.0
    assert PromptEnhancer._MAX_RETRIES == 1
    assert PromptEnhancer._RETRY_BACKOFF_SECONDS == 1.0


def test_classifier_timeout_and_attempt_constants():
    assert PromptClassifier._REQUEST_TIMEOUT_SECONDS == 10.0
    assert PromptClassifier._MAX_ATTEMPTS == 2
    assert PromptClassifier._RETRY_BACKOFF_SECONDS == 0.5


def test_enhancer_passes_timeout_to_groq():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_success_response()
    enhancer = PromptEnhancer(client=mock_client)

    enhancer._create_completion(
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.1,
        max_completion_tokens=500,
    )

    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs.get("timeout") == 15.0


def test_classifier_passes_timeout_to_groq():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_success_response('{"mode":"code"}')
    classifier = PromptClassifier(client=mock_client)

    classifier._create_completion("Write a Python script to sort a list")

    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs.get("timeout") == 10.0


def test_enhancer_retries_transient_error_up_to_max_limit():
    mock_client = MagicMock()
    # 2 consecutive transient errors (rate limit / timeout)
    mock_client.chat.completions.create.side_effect = [
        Exception("Rate limit 429: please slow down"),
        Exception("Rate limit 429: please slow down"),
    ]

    enhancer = PromptEnhancer(client=mock_client)

    with pytest.raises(Exception, match="Rate limit 429"):
        enhancer._create_completion(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.1,
            max_completion_tokens=200,
        )

    # Initial attempt + 1 retry = exactly 2 calls
    assert mock_client.chat.completions.create.call_count == 2


def test_enhancer_succeeds_after_one_transient_retry():
    mock_client = MagicMock()
    # 1 transient error followed by success
    mock_client.chat.completions.create.side_effect = [
        Exception("Connection timed out (timeout)"),
        _mock_success_response("Cleaned up prompt"),
    ]

    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer._create_completion(
        messages=[{"role": "user", "content": "Test"}],
        temperature=0.1,
        max_completion_tokens=200,
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert res.choices[0].message.content == "Cleaned up prompt"


def test_enhancer_does_not_retry_non_transient_error():
    mock_client = MagicMock()
    # Non-transient error (e.g. invalid auth / bad request)
    mock_client.chat.completions.create.side_effect = ValueError("Invalid argument")

    enhancer = PromptEnhancer(client=mock_client)

    with pytest.raises(ValueError, match="Invalid argument"):
        enhancer._create_completion(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.1,
            max_completion_tokens=200,
        )

    # Exactly 1 call (no retries for non-transient)
    assert mock_client.chat.completions.create.call_count == 1


def test_classifier_retries_transient_error_up_to_max_attempts():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("APIConnectionError: connection failed"),
        Exception("APIConnectionError: connection failed"),
    ]

    classifier = PromptClassifier(client=mock_client)

    with pytest.raises(Exception, match="APIConnectionError"):
        classifier._create_completion("Test prompt")

    # Exactly _MAX_ATTEMPTS = 2 calls
    assert mock_client.chat.completions.create.call_count == 2


def test_classifier_succeeds_on_second_attempt():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("503 Service Unavailable"),
        _mock_success_response('{"mode":"creative"}'),
    ]

    classifier = PromptClassifier(client=mock_client)
    content, tokens, truncated = classifier._create_completion("Write a poem")

    assert mock_client.chat.completions.create.call_count == 2
    assert content == '{"mode":"creative"}'
    assert truncated is False


def test_enhancer_surfaces_final_failure_cleanly():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("ReadTimeout: upstream server did not respond")

    enhancer = PromptEnhancer(client=mock_client)
    original_prompt = "Hello! Can you please help me write a python script to parse logs? Thank you!"
    result = enhancer.enhance(original_prompt, mode="code")

    # Error must be cleanly returned without unhandled crash
    assert result["success"] is False
    assert result["decision"] == "error"
    assert "ReadTimeout" in result["message"]
    assert result["enhanced_prompt"] == original_prompt
