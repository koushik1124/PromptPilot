"""
PromptPilot optimization gate.

Purpose:
    Decide deterministically whether a prompt deserves LLM calls
    (classifier + enhancer).

Design principles:
    - Do not spend LLM calls on prompts that are already sufficient.
    - Ambiguous prompts stay unchanged rather than being guessed at.
    - Already structured prompts stay unchanged.
    - Structured prompts whose line breaks were lost in transport
      (inline enumerations) are still recognized as structured.
    - Short direct tasks may bypass the LLM in explicit concise/code
      modes. Auto mode always reaches the classifier.
    - No semantic rewriting is performed here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Conversational / filler detection
# ---------------------------------------------------------------------------

_CONVERSATIONAL_PATTERNS = (
    re.compile(
        r"^\s*(?:hi|hello|hey)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please|kindly|thank you|thanks)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i was wondering if|could you please|would you please|"
        r"can you please|i would like you to|if you could)\b",
        re.IGNORECASE,
    ),
)


# ---------------------------------------------------------------------------
# Ambiguous prompts
# ---------------------------------------------------------------------------

_AMBIGUOUS_PATTERNS = (
    re.compile(
        r"^\s*(?:fix|debug|solve|change|improve|optimize)"
        r"\s+(?:this|it)\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:write|create|make|build)"
        r"\s+(?:code|something)\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:help me|help)\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*make\s+it\s+work\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:do|make|handle)"
        r"\s+this\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
    # "make this better" / "make it professional" — no recoverable
    # task; previously leaked to the classifier.
    re.compile(
        r"^\s*make\s+(?:this|it)\s+"
        r"(?:better|nicer|clearer|cleaner|professional|shorter|longer|"
        r"formal|simple|simpler)"
        r"\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
)


# ---------------------------------------------------------------------------
# Constraint detection
# ---------------------------------------------------------------------------

_CONSTRAINT_KEYWORDS_RE = re.compile(
    r"\b(?:must|should|need|needs|require|required|without|with|only|"
    r"exactly|at least|at most|minimum|maximum|use|using|avoid|never|"
    r"don't|do not|mustn't|shouldn't|deliver|deliverable|format|"
    r"return|output|constraint|deadline|limit)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Short direct task detection
# ---------------------------------------------------------------------------

_SIMPLE_DIRECT_TASK_PATTERN = re.compile(
    r"\s*(?:"
    r"make|create|write|build|generate|implement|add|remove|delete|"
    r"convert|parse|extract|find|calculate|sort|filter|summarize|"
    r"explain|translate|list|describe|compare|define|analyze|review|"
    r"outline|rewrite|draft|check|validate|plan|design|optimize|"
    r"rename|format|split|merge|show|tell|give|print"
    r")\b.+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(
    r"(?m)^\s*(?:"
    r"#{1,6}\s+.+"
    r"|[A-Z][A-Za-z0-9 /&_-]{2,40}:"
    r")\s*$"
)

_NUMBERED_ITEM_RE = re.compile(
    r"(?m)^\s*(?:\d+[.)]|[a-zA-Z][.)])\s+\S+"
)

_BULLET_ITEM_RE = re.compile(
    r"(?m)^\s*[-*•]\s+\S+"
)

# Inline enumerations: "1. Do X. 2. Do Y." written on a single line.
# Catches structured prompts whose line breaks were lost between the
# client and this service (transport-layer newline stripping).
#
# Requires BOTH an enumeration starting at 1 and continuing at 2, with
# a capital letter after each, so ordinary prose that happens to
# follow digits ("It costs 5. Do it.") does not false-positive.
# Capital letters also keep genuinely messy lowercase run-ons eligible
# for enhancement — the failure direction here is conservative.
_INLINE_LIST_START_RE = re.compile(
    r"(?:^|[\s.!?;:])1[.)]\s+[A-Z]"
)

_INLINE_LIST_CONT_RE = re.compile(
    r"(?:^|[\s.!?;:])2[.)]\s+[A-Z]"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_BYPASS_WORD_COUNT = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize whitespace for deterministic comparisons."""
    return re.sub(r"\s+", " ", text.strip())


def _word_count(text: str) -> int:
    """Return a simple whitespace-based word count."""
    return len(re.findall(r"\S+", text))


def _has_multiple_clauses(text: str) -> bool:
    """
    Detect whether the prompt contains multiple task clauses.

    Preserves single-phrase coordinate tasks (e.g., 'read csv and remove duplicates').
    """
    if re.search(
        r"\b(?:and then|then|also|as well as|while|plus)\b",
        text,
        re.IGNORECASE,
    ):
        return True

    if len(re.findall(r"\band\b", text, re.IGNORECASE)) >= 2:
        return True

    return len(re.findall(r"[,;]", text)) >= 2


def _has_conversational_filler(text: str) -> bool:
    """Return True when the prompt contains conversational filler."""
    normalized = _normalize(text)

    return any(
        pattern.search(normalized)
        for pattern in _CONVERSATIONAL_PATTERNS
    )


def _is_ambiguous(text: str) -> bool:
    """Return True when the prompt is too vague to safely enhance."""
    normalized = _normalize(text)

    return any(
        pattern.fullmatch(normalized)
        for pattern in _AMBIGUOUS_PATTERNS
    )


def _has_constraints(text: str) -> bool:
    """Return True when explicit constraints are present."""
    return bool(_CONSTRAINT_KEYWORDS_RE.search(text))


def _count_structure_markers(text: str) -> int:
    """Count obvious structural markers."""
    return (
        len(_SECTION_HEADER_RE.findall(text))
        + len(_NUMBERED_ITEM_RE.findall(text))
        + len(_BULLET_ITEM_RE.findall(text))
    )


def is_already_structured(text: str) -> bool:
    """
    Determine whether a prompt already has meaningful visible structure.

    A single section header is enough to indicate deliberate structure.
    Two numbered or bullet items also count as structured.

    Inline enumerations ("1. Do X. 2. Do Y." on one line) count as
    structured too — they are almost always structured prompts whose
    newlines were stripped in transport. Enhancing them would only
    re-add the line breaks that were lost, at token cost, while the
    content was already well organized.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    section_count = len(_SECTION_HEADER_RE.findall(text))
    numbered_count = len(_NUMBERED_ITEM_RE.findall(text))
    bullet_count = len(_BULLET_ITEM_RE.findall(text))

    inline_numbered = (
        bool(_INLINE_LIST_START_RE.search(text))
        and bool(_INLINE_LIST_CONT_RE.search(text))
    )

    return (
        section_count >= 1
        or numbered_count >= 2
        or bullet_count >= 2
        or inline_numbered
    )


# ---------------------------------------------------------------------------
# OptimizationGate
# ---------------------------------------------------------------------------

class OptimizationGate:
    """
    Deterministic pre-check for PromptPilot.

    The gate decides whether LLM calls are worth making.

    Important:
        Auto mode passes mode=None intentionally.

        Therefore the short-task bypass below applies only to explicit
        manual modes ("concise" and "code"), not Auto mode. Auto mode
        always reaches the classifier, which is itself route-only.
    """

    @staticmethod
    def is_already_structured(text: str) -> bool:
        """Expose structure detection through the class API."""
        return is_already_structured(text)

    @classmethod
    def evaluate(
        cls,
        prompt: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether a prompt should be sent for enhancement.

        Returns:
            {
                "should_optimize": bool,
                "is_ambiguous": bool,
                "is_conversational": bool,
                "is_already_structured": bool,
                "reason": str,
            }
        """

        # ---------------------------------------------------------------
        # Empty prompt
        # ---------------------------------------------------------------

        if not isinstance(prompt, str) or not prompt.strip():
            return {
                "should_optimize": False,
                "is_ambiguous": False,
                "is_conversational": False,
                "is_already_structured": False,
                "reason": "Prompt is empty; no optimization needed.",
            }

        original = prompt.strip()
        normalized = _normalize(original)

        words = _word_count(normalized)
        multiline = "\n" in original
        multiple_clauses = _has_multiple_clauses(normalized)
        has_constraints = _has_constraints(original)

        ambiguous = _is_ambiguous(normalized)
        conversational = _has_conversational_filler(original)
        structured = is_already_structured(original)

        # ---------------------------------------------------------------
        # Ambiguous prompt
        # ---------------------------------------------------------------

        if ambiguous:
            return {
                "should_optimize": False,
                "is_ambiguous": True,
                "is_conversational": conversational,
                "is_already_structured": structured,
                "reason": (
                    "Prompt is too ambiguous to enhance safely; "
                    "kept original."
                ),
            }

        # ---------------------------------------------------------------
        # Already structured (including inline enumerations)
        # ---------------------------------------------------------------

        if structured:
            return {
                "should_optimize": False,
                "is_ambiguous": False,
                "is_conversational": conversational,
                "is_already_structured": True,
                "reason": (
                    "Prompt is already clearly structured and specific; "
                    "no optimization needed."
                ),
            }

        # ---------------------------------------------------------------
        # Conversational prompt
        # ---------------------------------------------------------------

        if conversational:
            return {
                "should_optimize": True,
                "is_ambiguous": False,
                "is_conversational": True,
                "is_already_structured": False,
                "reason": (
                    "Prompt contains conversational filler; "
                    "optimization may improve clarity."
                ),
            }

        # ---------------------------------------------------------------
        # Short direct task bypass
        #
        # IMPORTANT:
        # Auto passes mode=None.
        #
        # Therefore only explicit concise/code modes can bypass here.
        # Auto mode always reaches the classifier.
        # ---------------------------------------------------------------

        if (
            mode in {"concise", "code", None}
            and words <= _MAX_BYPASS_WORD_COUNT
            and not has_constraints
            and not multiline
            and not multiple_clauses
            and _SIMPLE_DIRECT_TASK_PATTERN.fullmatch(normalized)
        ):
            return {
                "should_optimize": False,
                "is_ambiguous": False,
                "is_conversational": False,
                "is_already_structured": False,
                "reason": (
                    "Prompt is already a short, direct task with no "
                    "additional constraints; no optimization needed."
                ),
            }

        # ---------------------------------------------------------------
        # Default
        # ---------------------------------------------------------------

        return {
            "should_optimize": True,
            "is_ambiguous": False,
            "is_conversational": False,
            "is_already_structured": False,
            "reason": (
                "Prompt may benefit from optimization; "
                "LLM enhancement allowed."
            ),
        }