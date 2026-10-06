"""
Phase C Quality and Decision Intelligence Benchmark Suite.

Contains 20 deterministic benchmark cases evaluating:
1. Already-optimal prompts (NO_CHANGE bypass with zero overhead)
2. Prompts that genuinely benefit from STRUCTURE (Decomposition and clear formatting)
3. Clear prompts that should only receive REFINE or remain unchanged (Light touch, no structural inflation)
4. Verbose prompts that genuinely benefit from COMPRESS (Conversational/fluff removal)
5. Creative prompts that benefit from CREATIVE_EXPAND without invented constraints
6. Ambiguous prompts that must be bypassed rather than guessed at

Tests intent preservation, requirement retention, constraint/negation enforcement,
technical terms preservation, avoidance of unrequested expansion, and decision correctness.
"""

from unittest.mock import MagicMock
import pytest

from app.services.optimization_gate import OptimizationGate
from app.services.prompt_enhancer import PromptEnhancer
from app.services.strategy_router import EnhancementStrategy, route_strategy
from app.services.strategy_rules import get_strategy_rule


def _build_mock_response(content: str = "Enhanced prompt", total_tokens: int = 50):
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.total_tokens = total_tokens

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    return mock_resp


# =========================================================================
# Pillar 1: Already-Optimal Prompts (Zero Overhead / NO_CHANGE)
# =========================================================================

def test_p1_case1_fully_structured_microservice_spec():
    """Case 1: Fully structured specification with multiple sections and bullets."""
    prompt = """Design a Payment Gateway microservice in Go.

Requirements:
- Idempotency key validation via Redis
- Stripe and PayPal provider adapters
- Circuit breaker with exponential backoff

Constraints:
- Response latency under 50ms (p99)
- Do not store raw credit card numbers
"""
    # 1. Router must select NO_CHANGE
    assert route_strategy("code", prompt) == EnhancementStrategy.NO_CHANGE

    # 2. Gate must identify it as already structured
    assert OptimizationGate.is_already_structured(prompt) is True
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["should_optimize"] is False

    # 3. Enhancer must return original without calling LLM
    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt, mode="auto")
    assert res["success"] is True
    assert res["decision"] == "unchanged"
    assert res["enhancement_overhead_tokens"] == 0
    assert res["enhanced_prompt"] == prompt.strip()
    mock_client.chat.completions.create.assert_not_called()


def test_p1_case2_short_direct_task_sort_dict():
    """Case 2: Short, concise direct programming instruction."""
    prompt = "Sort a list of dictionaries by key 'timestamp' in Python."

    # Direct task bypasses gate
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt, mode="code")
    assert res["decision"] == "unchanged"
    assert res["enhancement_overhead_tokens"] == 0
    mock_client.chat.completions.create.assert_not_called()


def test_p1_case3_already_clean_csv_to_json():
    """Case 3: Clean minimal transformation command."""
    prompt = "Convert this CSV to JSON."

    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt, mode="concise")
    assert res["decision"] == "unchanged"
    assert res["enhanced_prompt"] == prompt
    mock_client.chat.completions.create.assert_not_called()


# =========================================================================
# Pillar 2: Prompts that Genuinely Benefit from STRUCTURE
# =========================================================================

def test_p2_case4_unformatted_s3_data_pipeline():
    """Case 4: Multi-step data processing workflow in a single run-on sentence."""
    prompt = (
        "Write a python script to download sales data from s3 bucket parse "
        "csv rows aggregate monthly totals by region and output summary to parquet file."
    )
    # Strategy should be STRUCTURE
    strategy = route_strategy("code", prompt)
    assert strategy == EnhancementStrategy.STRUCTURE

    structured_output = (
        "Write a Python script that performs the following steps:\n"
        "1. Downloads sales data from an S3 bucket.\n"
        "2. Parses CSV rows and handles missing records.\n"
        "3. Aggregates monthly totals grouped by region.\n"
        "4. Exports the summary dataset to a Parquet file."
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        structured_output, total_tokens=65,
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {"mode": "code", "tokens": 10, "success": True}

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert res["decision"] == "enhanced"
    assert res["strategy"] == EnhancementStrategy.STRUCTURE.value

    # Validate tech terms and deliverables preserved
    lowered = res["enhanced_prompt"].lower()
    for term in ["s3", "csv", "monthly totals", "region", "parquet"]:
        assert term in lowered, f"Term '{term}' missing from structured output"


def test_p2_case5_compound_multi_tier_fullstack_spec():
    """Case 5: Multi-sentence sequential fullstack specification."""
    prompt = (
        "I need a full stack app for an e-commerce store. "
        "First build a React frontend with a product catalog, shopping cart, and Stripe checkout modal. "
        "Then create an Express backend with endpoints for product search, order placement, and user auth. "
        "Also set up a PostgreSQL database with schema migrations. "
        "Finally add Docker compose to run everything with Redis caching."
    )
    # Sequence markers ('first', 'then', 'finally') force STRUCTURE
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


def test_p2_case6_unformatted_rest_api_with_negative_constraint():
    """Case 6: Unformatted Go REST API specification with positive & negative constraints."""
    prompt = (
        "Build a REST API in Go with Gin and GORM for managing books with title author and isbn. "
        "Must include pagination. Do not use global database variables."
    )
    structured_output = (
        "Build a REST API in Go for managing books (fields: title, author, isbn).\n\n"
        "Technical Stack:\n"
        "- Framework: Gin\n"
        "- ORM: GORM\n\n"
        "Requirements:\n"
        "- Implement pagination for list endpoints\n\n"
        "Constraints:\n"
        "- Do not use global database variables"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        structured_output, total_tokens=70,
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {"mode": "code", "tokens": 12, "success": True}

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert res["decision"] == "enhanced"
    lowered = res["enhanced_prompt"].lower()
    for term in ["gin", "gorm", "isbn", "pagination"]:
        assert term in lowered, f"Term '{term}' missing from enhanced output"
    assert "do not" in lowered or "avoid" in lowered, "Negation missing"
    assert "global database" in lowered or "global" in lowered, "Constraint subject missing"


def test_p2_case7_multi_action_data_analysis():
    """Case 7: Multi-action data analysis prompt needing structured steps."""
    prompt = "Load sales data from sqlite database clean missing records compute moving average of revenue and generate a matplotlib chart."
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


# =========================================================================
# Pillar 3: Clear Prompts that Should Only Receive REFINE / Light Touch
# =========================================================================

def test_p3_case8_fastapi_benchmark_light_refine():
    """Case 8: Multi-sentence clear prose with negative constraint (FastAPI benchmark)."""
    prompt = (
        "Build a FastAPI backend for a task management application "
        "using PostgreSQL, SQLAlchemy, Pydantic, and JWT authentication. "
        "Do not use MongoDB."
    )
    # Must route to REFINE (not bloated STRUCTURE)
    assert route_strategy("code", prompt) == EnhancementStrategy.REFINE


def test_p3_case9_react_dashboard_benchmark_refine():
    """Case 9: 2-sentence UI request (React dashboard benchmark)."""
    prompt = (
        "Create a React dashboard for monitoring cloud servers. "
        "It should display CPU usage, memory consumption, and network traffic in real-time charts."
    )
    # Must route to REFINE
    assert route_strategy("code", prompt) == EnhancementStrategy.REFINE


def test_p3_case10_clean_instruction_with_minor_grammar():
    """Case 10: Clear technical prompt with minor phrasing polish."""
    prompt = (
        "Write a Python function that calculate the GCD of two numbers using Euclidean algorithm. "
        "Include docstring and type hints."
    )
    assert route_strategy("code", prompt) == EnhancementStrategy.REFINE

    refined_output = (
        "Write a Python function that calculates the greatest common divisor (GCD) of two numbers "
        "using the Euclidean algorithm. Include a docstring and type hints."
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(refined_output, total_tokens=40)
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {"mode": "code", "tokens": 10, "success": True}

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert res["decision"] == "enhanced"
    assert res["strategy"] == EnhancementStrategy.REFINE.value
    # Ensure no unrequested tests or CLI frameworks were injected
    assert "pytest" not in res["enhanced_prompt"].lower()
    assert "argparse" not in res["enhanced_prompt"].lower()


def test_p3_case11_partially_structured_section_header():
    """Case 11: Prompt with 1 section header routes to REFINE to avoid re-structuring."""
    prompt = "Build a FastAPI backend.\n\nRequirements:\nUse PostgreSQL and SQLAlchemy."
    assert route_strategy("code", prompt) == EnhancementStrategy.REFINE


# =========================================================================
# Pillar 4: Verbose Prompts that Genuinely Benefit from COMPRESS
# =========================================================================

def test_p4_case12_conversational_markdown_to_html():
    """Case 12: Polite conversational filler requiring compression."""
    prompt = (
        "Hello there! I was wondering if you could please kindly help me by writing "
        "a Python script to convert Markdown text to HTML format? Thank you so much!"
    )
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["is_conversational"] is True
    assert gate_res["should_optimize"] is True

    assert route_strategy("concise", prompt) == EnhancementStrategy.COMPRESS

    compressed_output = "Write a Python script to convert Markdown text to HTML format."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(compressed_output, total_tokens=25)

    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt, mode="concise", bypass_gate=True)

    assert res["decision"] == "enhanced"
    assert res["tokens_saved"] > 0
    lowered = res["enhanced_prompt"].lower()
    assert "python" in lowered
    assert "markdown" in lowered
    assert "html" in lowered
    assert "hello" not in lowered
    assert "please" not in lowered


def test_p4_case13_redundant_preamble_in_concise_mode():
    """Case 13: Redundant preamble in concise mode."""
    prompt = "Could you please take a look at this and write a short summary of this article in bullet points?"
    assert route_strategy("concise", prompt) == EnhancementStrategy.COMPRESS


def test_p4_case14_wordy_email_request():
    """Case 14: Verbose email drafting request."""
    prompt = "Hi assistant, I would really appreciate it if you could write an email to my manager asking for time off next Friday."
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["is_conversational"] is True
    assert gate_res["should_optimize"] is True


# =========================================================================
# Pillar 5: Creative Prompts (CREATIVE_EXPAND Without Invented Constraints)
# =========================================================================

def test_p5_case15_creative_astronaut_story():
    """Case 15: Underspecified story prompt expands dimensions without invented constraints."""
    prompt = "Write a short story about an astronaut stranded on Mars."
    assert route_strategy("creative", prompt) == EnhancementStrategy.CREATIVE_EXPAND

    expanded_output = (
        "Write a short science fiction story about an astronaut stranded on Mars.\n"
        "Develop the protagonist's survival struggle, psychological state, and environmental obstacles, "
        "leading toward a compelling resolution."
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(expanded_output, total_tokens=45)
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {"mode": "creative", "tokens": 8, "success": True}

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert res["decision"] == "enhanced"
    assert res["strategy"] == EnhancementStrategy.CREATIVE_EXPAND.value
    lowered = res["enhanced_prompt"].lower()
    assert "astronaut" in lowered
    assert "mars" in lowered


def test_p5_case16_creative_saas_brainstorming():
    """Case 16: Creative startup ideation with domain focus."""
    prompt = "Generate 5 unique SaaS business ideas combining computer vision and agriculture."
    assert route_strategy("creative", prompt) == EnhancementStrategy.CREATIVE_EXPAND


def test_p5_case17_creative_worldbuilding_with_prohibition():
    """Case 17: Creative worldbuilding with explicit prohibitions (negations must survive)."""
    prompt = "Create a fantasy world with magic based on sound, but do not include elves or dwarves."
    assert route_strategy("creative", prompt) == EnhancementStrategy.CREATIVE_EXPAND

    expanded_output = (
        "Design an original fantasy world where magic is powered by sound and acoustic resonance.\n"
        "Flesh out the magic system mechanics, societal impact, and key factions.\n"
        "Constraint: Do not include elves or dwarves."
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(expanded_output, total_tokens=55)
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {"mode": "creative", "tokens": 10, "success": True}

    enhancer = PromptEnhancer(client=mock_client, classifier=mock_classifier)
    res = enhancer.enhance(prompt, mode="auto", bypass_gate=True)

    assert res["decision"] == "enhanced"
    lowered = res["enhanced_prompt"].lower()
    assert "sound" in lowered or "magic" in lowered
    assert "elves" in lowered and "dwarves" in lowered
    assert "do not" in lowered or "avoid" in lowered, "Prohibition missing"


# =========================================================================
# Pillar 6: Ambiguous Prompts (Bypassed / Not Hallucinated)
# =========================================================================

def test_p6_case18_ambiguous_fix_this():
    """Case 18: Ultra-vague prompt 'fix this' must be blocked by ambiguity check."""
    prompt = "fix this"
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["is_ambiguous"] is True
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt)
    assert res["decision"] == "unchanged"
    assert res["enhanced_prompt"] == prompt
    mock_client.chat.completions.create.assert_not_called()


def test_p6_case19_ambiguous_write_code():
    """Case 19: Vague generative prompt 'write code' must be blocked."""
    prompt = "write code"
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["is_ambiguous"] is True
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt)
    assert res["decision"] == "unchanged"
    mock_client.chat.completions.create.assert_not_called()


def test_p6_case20_ambiguous_make_it_better():
    """Case 20: Vague improvement prompt 'make it better' must be blocked."""
    prompt = "make it better"
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["is_ambiguous"] is True
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    res = enhancer.enhance(prompt)
    assert res["decision"] == "unchanged"
    mock_client.chat.completions.create.assert_not_called()


# =========================================================================
# REFINE Acceptance Boundary Tests (+6, +7, +8, +9 token growth)
# =========================================================================

class TestRefineBoundaryAcceptance:
    """Deterministic boundary tests for REFINE growth tolerance without structural markers."""

    PROSE_ORIG = "Write a Python function that calculate the GCD of two numbers using Euclidean algorithm."
    PROSE_ENH = "Write a Python function that calculates the greatest common divisor (GCD) of two numbers using Euclidean algorithm."

    def test_refine_plus_6_tokens_accepted(self):
        """+6 tokens growth with REFINE and no structure → ACCEPTED."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=56,  # +6 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert accept is True
        assert "within tolerance" in reason

    def test_refine_plus_7_tokens_behavior(self):
        """+7 tokens growth with REFINE and no structure → ACCEPTED under current tolerance (8)."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=57,  # +7 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert accept is True
        assert "within tolerance" in reason

    def test_refine_plus_8_tokens_behavior(self):
        """+8 tokens growth with REFINE and no structure → ACCEPTED at current exact bound (8)."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=58,  # +8 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert accept is True
        assert "within tolerance" in reason

    def test_refine_plus_9_tokens_rejected(self):
        """+9 tokens growth with REFINE and no structure → REJECTED (exceeds tolerance)."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=59,  # +9 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert accept is False
        assert "without introducing meaningful structure" in reason

    def test_refine_larger_growth_rejected(self):
        """+15 tokens growth with REFINE and no structure → REJECTED."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=65,  # +15 tokens
            strategy=EnhancementStrategy.REFINE,
        )
        assert accept is False
        assert "without introducing meaningful structure" in reason

    def test_non_refine_growth_without_structure_rejected(self):
        """+4 tokens growth under STRUCTURE strategy without actual structure markers → REJECTED."""
        accept, reason = PromptEnhancer._should_accept_rewrite(
            mode="code",
            original=self.PROSE_ORIG,
            enhanced=self.PROSE_ENH,
            original_tokens=50,
            enhanced_tokens=54,  # +4 tokens
            strategy=EnhancementStrategy.STRUCTURE,
        )
        assert accept is False
        assert "without introducing meaningful structure" in reason
