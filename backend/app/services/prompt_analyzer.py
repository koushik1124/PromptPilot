"""
Prompt Analyzer Service
Analyzes prompt directness, specificity, constraint density, and ambiguity.
"""

import re
from typing import Dict, Any, List

_IMPERATIVE_START = re.compile(
    r"^\s*(?:create|build|generate|write|implement|add|remove|delete|refactor|extract|parse|convert|format|find|calculate|compare|validate|filter|sort|optimize|fix|design|develop|plan|structure|outline|summarize|explain|describe|draft|review|setup|configure)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL_FLUFF = re.compile(
    r"\b(?:hello|hi|please\s+help|could\s+you\s+please|i\s+was\s+wondering|as\s+an\s+ai)\b",
    re.IGNORECASE,
)


class PromptAnalyzer:
    def analyze(self, prompt: str) -> Dict[str, Any]:
        text = prompt.strip()
        words = text.split()
        word_count = len(words)

        if not text:
            return {
                "score": 0,
                "word_count": 0,
                "is_direct": False,
                "has_constraints": False,
                "has_fluff": False,
                "suggestions": ["Prompt is empty."],
            }

        is_direct = bool(_IMPERATIVE_START.search(text))
        has_fluff = bool(_CONVERSATIONAL_FLUFF.search(text))
        has_constraints = bool(
            re.search(
                r"\b(?:must|should|do not|don't|using|with|return|format|schema)\b",
                text,
                re.IGNORECASE,
            )
        )
        has_context = word_count >= 8

        score = 0
        suggestions: List[str] = []

        # Direct imperative start (+30)
        if is_direct:
            score += 30
        else:
            suggestions.append("Start directly with an action verb (e.g., 'Build', 'Extract', 'Calculate').")

        # Free of conversational fluff (+30)
        if not has_fluff:
            score += 30
        else:
            suggestions.append("Remove conversational greetings and polite filler to save tokens.")

        # Has explicit constraints/parameters (+25)
        if has_constraints:
            score += 25
        else:
            suggestions.append("Specify explicit constraints or expected inputs/outputs.")

        # Has sufficient detail/context (+15)
        if has_context:
            score += 15
        else:
            suggestions.append("Provide sufficient detail for deterministic task execution.")

        return {
            "score": score,
            "word_count": word_count,
            "is_direct": is_direct,
            "has_constraints": has_constraints,
            "has_fluff": has_fluff,
            "suggestions": suggestions,
        }