from app.services.strategy_router import (
    EnhancementStrategy,
    route_strategy,
)


def test_concise_routes_to_compress():
    result = route_strategy(
        "concise",
        "Write a Python function that removes duplicate CSV rows.",
    )

    assert result == EnhancementStrategy.COMPRESS


def test_creative_routes_to_creative_expand():
    result = route_strategy(
        "creative",
        "Give me practical AI startup ideas for developers.",
    )

    assert result == EnhancementStrategy.CREATIVE_EXPAND


def test_unstructured_code_routes_to_structure():
    result = route_strategy(
        "code",
        "Build a FastAPI backend with PostgreSQL and JWT authentication.",
    )

    assert result == EnhancementStrategy.STRUCTURE


def test_partially_structured_prompt_routes_to_refine():
    result = route_strategy(
        "code",
        """Build a FastAPI backend.

Requirements:
Use PostgreSQL and SQLAlchemy.
""",
    )

    assert result == EnhancementStrategy.REFINE


def test_already_structured_prompt_routes_to_no_change():
    result = route_strategy(
        "code",
        """Build a FastAPI backend.

Requirements:
- Use PostgreSQL
- Use SQLAlchemy

Constraints:
- Do not use MongoDB
""",
    )

    assert result == EnhancementStrategy.NO_CHANGE


def test_multi_sentence_prose_routes_to_refine():
    result = route_strategy(
        "code",
        "Create a React dashboard for monitoring cloud servers. It should display CPU usage, memory consumption, and network traffic in real-time charts.",
    )

    assert result == EnhancementStrategy.REFINE


def test_multi_sentence_fastapi_routes_to_refine():
    result = route_strategy(
        "code",
        "Build a FastAPI backend for a task management application using PostgreSQL, SQLAlchemy, Pydantic, and JWT authentication. Do not use MongoDB.",
    )

    assert result == EnhancementStrategy.REFINE


# =========================================================================
# Boundary Hardening Unit Tests (Phase B)
# =========================================================================


# --- Boundary 1: NO_CHANGE vs REFINE ---
def test_boundary_no_change_vs_refine_fully_structured():
    """A prompt with multiple structural markers (bullets + constraints) should remain NO_CHANGE."""
    prompt = """Implement a rate limiter middleware for FastAPI.

Features:
- Token bucket algorithm
- Redis backend for distributed state

Constraints:
- Return HTTP 429 when quota exceeded
- Max latency overhead under 2ms
"""
    assert route_strategy("code", prompt) == EnhancementStrategy.NO_CHANGE


def test_boundary_no_change_vs_refine_partially_structured():
    """A prompt with only 1 structural marker needs REFINE, not NO_CHANGE."""
    prompt = """Implement a rate limiter middleware for FastAPI.

Features:
Use token bucket algorithm with Redis backend and return 429 when exceeded.
"""
    assert route_strategy("code", prompt) == EnhancementStrategy.REFINE


# --- Boundary 2: REFINE vs STRUCTURE ---
def test_boundary_refine_vs_structure_single_sentence_unstructured():
    """A single unstructured sentence with multiple requirements needs STRUCTURE."""
    prompt = "Write a python script to parse logs filter errors calculate error rates and send an email alert."
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


def test_boundary_refine_vs_structure_clean_statement():
    """A simple self-contained single instruction with no structural clutter."""
    prompt = "Write a Python function to calculate the Haversine distance between two coordinates."
    # Single sentence without structure markers routes to STRUCTURE currently
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


def test_boundary_refine_vs_structure_adversarial_compound_multi_sentence():
    """Adversarial case: A genuinely compound, multi-tier request that should require

    STRUCTURE because it specifies multiple phases, components, inputs/outputs,
    and workflows dumped in prose across multiple sentences without formatting.
    """
    prompt = (
        "I need a full stack app for an e-commerce store. "
        "First build a React frontend with a product catalog, shopping cart, and Stripe checkout modal. "
        "Then create an Express backend with endpoints for product search, order placement, and user auth. "
        "Also set up a PostgreSQL database with schema migrations. "
        "Finally add Docker compose to run everything with Redis caching."
    )
    # Expected ideal behavior: Needs STRUCTURE to organize multi-tier architecture and steps.
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


# --- Boundary 3: STRUCTURE vs NO_CHANGE ---
def test_boundary_structure_vs_no_change_messy_spec():
    """A complex messy spec with no markdown structure requires STRUCTURE, not NO_CHANGE."""
    prompt = "Create a web scraper in Python using BeautifulSoup to extract product names prices ratings and stock status from e-commerce listings and export to CSV."
    assert route_strategy("code", prompt) == EnhancementStrategy.STRUCTURE


def test_boundary_structure_vs_no_change_well_structured_spec():
    """The same spec organized with sections and bullets is NO_CHANGE."""
    prompt = """Create a web scraper in Python using BeautifulSoup.

Requirements:
- Extract product names, prices, ratings, and stock status
- Export output to CSV
"""
    assert route_strategy("code", prompt) == EnhancementStrategy.NO_CHANGE


# --- Boundary 4: COMPRESS vs REFINE ---
def test_boundary_compress_vs_refine_mode_concise():
    """In concise mode, wordy requests route to COMPRESS."""
    prompt = "Could you please be so kind as to write a Python function that computes the factorial of a given integer number?"
    assert route_strategy("concise", prompt) == EnhancementStrategy.COMPRESS


def test_boundary_compress_vs_refine_mode_concise_already_structured():
    """In concise mode, an already fully structured prompt is preserved as NO_CHANGE."""
    prompt = """Compute factorial in Python.

Steps:
1. Handle negative inputs
2. Handle base cases 0 and 1
3. Compute iterative factorial
"""
    assert route_strategy("concise", prompt) == EnhancementStrategy.NO_CHANGE


# --- Boundary 5: CREATIVE_EXPAND vs REFINE ---
def test_boundary_creative_expand_vs_refine_mode_creative():
    """In creative mode, underspecified creative requests route to CREATIVE_EXPAND."""
    prompt = "Write a short sci-fi story about a sentient lighthouse on a distant ocean planet."
    assert route_strategy("creative", prompt) == EnhancementStrategy.CREATIVE_EXPAND


def test_boundary_creative_expand_vs_refine_mode_code():
    """In code mode, a technical prompt does not route to CREATIVE_EXPAND."""
    prompt = "Write a shader in GLSL to simulate realistic ocean waves."
    assert route_strategy("code", prompt) != EnhancementStrategy.CREATIVE_EXPAND
