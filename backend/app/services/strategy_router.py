import re
from enum import Enum


class EnhancementStrategy(str, Enum):
    NO_CHANGE = "NO_CHANGE"
    COMPRESS = "COMPRESS"
    STRUCTURE = "STRUCTURE"
    REFINE = "REFINE"
    CREATIVE_EXPAND = "CREATIVE_EXPAND"


def _count_structure_markers(prompt: str) -> int:
    """Count lightweight structural signals without using an LLM."""
    markers = 0

    lines = prompt.splitlines()

    # Numbered requirements
    if any(line.lstrip()[:2].rstrip(".").isdigit() for line in lines):
        markers += 1

    # Bullets
    if any(line.lstrip().startswith(("-", "*", "•")) for line in lines):
        markers += 1

    # Common section/requirement labels
    structural_terms = (
        "requirements:",
        "constraints:",
        "requirements",
        "constraints",
        "steps:",
        "features:",
        "technical requirements:",
        "acceptance criteria:",
    )

    lowered = prompt.lower()
    if any(term in lowered for term in structural_terms):
        markers += 1

    return markers


def _is_already_structured(prompt: str) -> bool:
    """Return True when the prompt already has meaningful organization."""
    return _count_structure_markers(prompt) >= 2


_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")

_SEQUENCE_MARKERS = (
    r"\bfirst\b",
    r"\bthen\b",
    r"\bnext\b",
    r"\bfinally\b",
    r"\bafter that\b",
    r"\bstep \d+\b",
)


def _count_sentences(text: str) -> int:
    """Count sentence-ending punctuation as a proxy for distinct statements."""
    return len(_SENTENCE_END_RE.findall(text))


def _has_sequence_markers(prompt: str) -> bool:
    """Detect if multi-sentence prose contains sequential step markers."""
    lowered = prompt.lower()
    return sum(1 for pattern in _SEQUENCE_MARKERS if re.search(pattern, lowered)) >= 2


def route_strategy(
    mode: str | None,
    prompt: str,
    gate_result=None,
) -> EnhancementStrategy:
    """
    Deterministically select the transformation strategy.

    This function must not call an LLM.
    """

    # Already well structured: don't unnecessarily rewrite it.
    if _is_already_structured(prompt):
        return EnhancementStrategy.NO_CHANGE

    if mode == "concise":
        return EnhancementStrategy.COMPRESS

    if mode == "creative":
        return EnhancementStrategy.CREATIVE_EXPAND

    # A partially structured technical/detailed request
    # needs refinement rather than a full restructuring.
    if _count_structure_markers(prompt) >= 1:
        return EnhancementStrategy.REFINE

    # A clear multi-sentence prompt (>= 2 distinct statements) without
    # explicit sequential step markers is already organized prose that
    # needs at most minor wording polish, not structural reorganization.
    if _count_sentences(prompt) >= 2 and not _has_sequence_markers(prompt):
        return EnhancementStrategy.REFINE

    return EnhancementStrategy.STRUCTURE