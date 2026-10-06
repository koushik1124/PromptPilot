"""Token counting utilities — approximate (cl100k_base), model-agnostic."""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder.encode(text))


def format_token_change(original_tokens: int, enhanced_tokens: int) -> str:
    """
    Format neutral, human-readable token change string.

    SHORTER: 'X fewer tokens (-Y%)'
    SAME:    'No change (0%)'
    LONGER:  'X more tokens (+Y%)'
    """
    token_delta = enhanced_tokens - original_tokens
    if original_tokens == 0:
        if enhanced_tokens == 0:
            return "No change (0%)"
        return f"{enhanced_tokens} more tokens (+100.0%)"

    pct = round((token_delta / original_tokens) * 100, 1)
    if token_delta < 0:
        return f"{abs(token_delta)} fewer tokens ({pct}%)"
    elif token_delta == 0:
        return "No change (0%)"
    else:
        return f"{token_delta} more tokens (+{pct}%)"


def calculate_token_metrics(original_tokens: int, enhanced_tokens: int) -> dict:
    """
    Calculate neutral token change metrics alongside legacy tokens_saved accounting.

    token_delta: enhanced_tokens - original_tokens (<0: shorter, 0: same, >0: longer)
    token_change_percent: ((enhanced_tokens - original_tokens) / original_tokens) * 100
    tokens_saved: original_tokens - enhanced_tokens (internal accounting)
    """
    token_delta = enhanced_tokens - original_tokens
    token_change_percent = (
        round((token_delta / original_tokens) * 100, 1)
        if original_tokens
        else 0.0
    )

    tokens_saved = original_tokens - enhanced_tokens
    percent_saved = (
        round((tokens_saved / original_tokens * 100), 1)
        if original_tokens
        else 0.0
    )

    return {
        "original_tokens": original_tokens,
        "enhanced_tokens": enhanced_tokens,
        "token_delta": token_delta,
        "token_change_percent": token_change_percent,
        "token_change_display": format_token_change(original_tokens, enhanced_tokens),
        "tokens_saved": tokens_saved,
        "percent_saved": percent_saved,
    }


def compare(original: str, enhanced: str) -> dict:
    orig = count_tokens(original)
    enh = count_tokens(enhanced)
    return calculate_token_metrics(orig, enh)