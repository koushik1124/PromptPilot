STRATEGY_RULES = {
    "COMPRESS": (
        "Remove filler, redundancy, and conversational padding. "
        "Preserve every requirement, constraint, exclusion, named technology, "
        "format, and deliverable. Do not add content. Return only the "
        "compressed prompt."
    ),

    "STRUCTURE": (
        "Clarify the request with minimal restructuring when useful. "
        "Prefer compact bulleted or numbered lists over labeled categories. "
        "Preserve the user's wording and all requirements, constraints, "
        "and exclusions. Do not add requirements, assumptions, examples, "
        "technologies, constraints, explanations, or categorization. "
        "Return only the structured prompt."
    ),

    "REFINE": (
        "Make only minor wording or clarity improvements. Preserve the "
        "existing sentence structure, ordering, and formatting. Do not "
        "convert prose into lists, headings, labels, or categories. Do not "
        "add, remove, or reorganize content. Return only the refined prompt."
    ),

    "CREATIVE_EXPAND": (
        "Strengthen the creative request using only the minimum missing "
        "context needed to make it more actionable. Preserve the user's "
        "intent, tone, style, and freedom. Do not invent requirements, "
        "evaluation criteria, workflows, output formats, audience, genre, "
        "or other constraints. Return only the enhanced prompt."
    ),
}


def get_strategy_rule(strategy: str) -> str:
    return STRATEGY_RULES[strategy]
