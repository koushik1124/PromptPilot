CORE_RULES = (
    "Preserve intent, requirements, constraints, exclusions, "
    "named technologies/tools, formats, and deliverables. "
    "Never invent requirements or assumptions. "
    "Prefer the minimum wording needed for clarity. "
    "Return only the enhanced prompt."
)

MODE_RULES = {
    "concise": (
        "Remove filler and redundancy without losing meaning. "
        "Do not grow the prompt. Preserve code, data, URLs, quotes, "
        "and prohibitions."
    ),

    "detailed": (
        "Clarify and structure messy requests using short sections "
        "or numbered requirements when useful. "
        "Do not expand a clear request just to add explanation, "
        "examples, or detail. "
        "Add wording only when it materially improves clarity. "
        "Leave already-clear structured prompts unchanged."
    ),

    "code": (
        "Improve technical clarity and structure when useful. "
        "Prefer compact restructuring over verbose categorization. "
        "Do not turn existing technology lists into labeled explanations "
        "unless this materially improves execution. "
        "Preserve the existing stack, APIs, databases, requirements, "
        "constraints, and error handling. "
        "Do not add unrequested tests, authentication, Docker, "
        "architecture, deployment, or technologies."
    ),

    "creative": (
        "Strengthen creative prompts when useful while preserving the user's "
        "intent, tone, style, and freedom. "
        "Do not invent requirements, evaluation criteria, workflows, "
        "technical directions, or other constraints. "
        "Do not impose an unrequested genre, audience, aesthetic, or direction. "
        "For short prompts, add only the minimum missing creative context "
        "needed to make the request more actionable."
    ),
}