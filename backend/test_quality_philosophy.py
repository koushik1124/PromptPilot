"""
Test Suite for PromptPilot Quality-First Enhancement Philosophy

Tests the core principles:
1. Concise mode shortens redundant prompts.
2. Already concise prompts remain unchanged.
3. Detailed mode allows longer prompts when structure improves clarity.
4. Creative mode allows longer prompts when organizing creative dimensions.
5. Code mode remains unchanged when technical prompt is already strong.
6. Shorter output that loses requirements or prohibitions is rejected/reverted.
7. Longer output that exceeds growth ceilings (invented requirements) is rejected/reverted.
8. Longer output that preserves intent and improves clarity is accepted.
9. Prompt with no meaningful improvement returns the original.
10. Token statistics are accurately tracked across positive, zero, and negative token savings.
"""

from unittest.mock import MagicMock
import pytest

from app.services.prompt_enhancer import PromptEnhancer
from app.services.prompt_evaluator import PromptEvaluator
from app.services.optimization_gate import OptimizationGate


def _mock_groq(content: str, total_tokens: int = 200):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    if total_tokens is not None:
        mock_usage = MagicMock()
        mock_usage.total_tokens = total_tokens
        mock_response.usage = mock_usage
    else:
        mock_response.usage = None

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# =========================================================================
# 1. Concise prompt can become shorter when redundant (TEST A)
# =========================================================================
def test_concise_mode_shortens_redundant_prompt():
    original = "Hey, could you please convert this CSV file into JSON format for me?"
    enhanced = "Convert this CSV to JSON."
    
    mock_client = _mock_groq(content=enhanced, total_tokens=150)
    enhancer = PromptEnhancer(client=mock_client)
    evaluator = PromptEvaluator(client=mock_client)

    result = enhancer.enhance(original, mode="concise")

    assert result["success"] is True
    assert result["decision"] == "enhanced"
    assert result["enhanced_prompt"] == enhanced
    assert result["tokens_saved"] > 0
    assert result["enhanced_tokens"] < result["original_tokens"]

    eval_res = evaluator.evaluate(original, result["enhanced_prompt"], result["enhancement_overhead_tokens"])
    assert eval_res["constraints_likely_preserved"] is True


# =========================================================================
# 2. Already concise prompt can remain unchanged
# =========================================================================
def test_already_concise_prompt_remains_unchanged():
    prompt = "Convert this CSV to JSON."
    
    # Gate evaluates direct task without fluff as not needing optimization
    gate_res = OptimizationGate.evaluate(prompt)
    assert gate_res["should_optimize"] is False

    mock_client = MagicMock()
    enhancer = PromptEnhancer(client=mock_client)
    result = enhancer.enhance(prompt, mode="concise")

    mock_client.chat.completions.create.assert_not_called()
    assert result["decision"] == "unchanged"
    assert result["enhanced_prompt"] == prompt
    assert result["tokens_saved"] == 0


# =========================================================================
# 3. Detailed mode is allowed to produce a longer prompt (TEST C)
# =========================================================================
def test_detailed_mode_allows_longer_prompt_with_structure():
    original = (
        "I want to build a student productivity web application where users can create "
        "daily study schedules, add assignments with deadlines, mark tasks as completed, "
        "and see their weekly progress. I want the application to have a clean responsive "
        "interface and a backend that stores user data. Please suggest a practical "
        "architecture and development plan."
    )
    structured_enhanced = (
        "Design and plan a student productivity web application with the following requirements:\n\n"
        "Functional Requirements:\n"
        "1. Daily Study Schedules: Allow users to create and customize daily study schedules.\n"
        "2. Assignment Tracking: Add assignments with deadlines and track completion status.\n"
        "3. Progress Visualization: Display weekly progress reports and completion metrics.\n"
        "4. Responsive UI: Clean, responsive user interface.\n"
        "5. Backend & Storage: Persistent backend to securely store user data.\n\n"
        "Deliverables:\n"
        "- Practical architecture overview (frontend, backend, database)\n"
        "- Step-by-step development roadmap"
    )

    mock_client = _mock_groq(content=structured_enhanced, total_tokens=280)
    enhancer = PromptEnhancer(client=mock_client)
    evaluator = PromptEvaluator(client=mock_client)

    result = enhancer.enhance(original, mode="detailed")

    assert result["success"] is True
    assert result["decision"] == "enhanced"
    assert result["enhanced_prompt"] == structured_enhanced
    # Enhanced prompt is longer and uses more tokens
    assert result["enhanced_tokens"] > result["original_tokens"]
    assert result["tokens_saved"] < 0
    assert "added" in result["reason"]

    eval_res = evaluator.evaluate(original, result["enhanced_prompt"], result["enhancement_overhead_tokens"])
    assert eval_res["constraints_likely_preserved"] is True
    assert eval_res["estimated_effectiveness"] == "High (Quality & Structure Enhanced)"


# =========================================================================
# 4. Creative mode is allowed to produce a longer prompt (TEST D)
# =========================================================================
def test_creative_mode_allows_longer_prompt():
    original = (
        "I am designing a landing page for an AI-powered personal knowledge assistant. "
        "I want the page to make the product feel intelligent, futuristic, trustworthy, "
        "and easy to use. Give me some creative ideas for the visual direction, hero section, "
        "animations, and overall user experience."
    )
    creative_enhanced = (
        "Develop creative concepts for a landing page for an AI-powered personal knowledge assistant.\n\n"
        "Brand Attributes & Tone:\n"
        "- Intelligent, futuristic, trustworthy, and intuitive\n\n"
        "Exploration Areas:\n"
        "1. Visual Direction: Color palette, typography, layout aesthetic, and spatial design.\n"
        "2. Hero Section: Hook, headline structure, interactive demo concepts, and CTA placement.\n"
        "3. Animations & Micro-interactions: Motion principles that reinforce intelligence and speed.\n"
        "4. User Experience: Seamless user journey highlighting knowledge synthesis and privacy."
    )

    mock_client = _mock_groq(content=creative_enhanced, total_tokens=250)
    enhancer = PromptEnhancer(client=mock_client)
    evaluator = PromptEvaluator(client=mock_client)

    result = enhancer.enhance(original, mode="creative")

    assert result["success"] is True
    assert result["decision"] == "enhanced"
    assert result["enhanced_tokens"] > result["original_tokens"]
    assert result["tokens_saved"] < 0

    eval_res = evaluator.evaluate(original, result["enhanced_prompt"], result["enhancement_overhead_tokens"])
    assert eval_res["constraints_likely_preserved"] is True
    assert eval_res["estimated_effectiveness"] == "High (Quality & Structure Enhanced)"


# =========================================================================
# 5. Code mode remains unchanged when technical prompt is already strong (TEST B)
# =========================================================================
def test_code_mode_remains_unchanged_when_strong():
    original = (
        "Build a FastAPI endpoint that accepts a CSV file upload, validates that it contains "
        "name, email, and age columns, removes duplicate email addresses, stores the cleaned "
        "records in PostgreSQL, and returns the number of inserted records. Include proper "
        "error handling."
    )

    # When the model determines the technical prompt is already strong and returns it
    mock_client = _mock_groq(content=original, total_tokens=120)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(original, mode="code")

    assert result["success"] is True
    assert result["decision"] == "unchanged"
    assert result["enhanced_prompt"] == original
    assert result["tokens_saved"] == 0


# =========================================================================
# 6. Shorter output that loses requirements/negations must be rejected
# =========================================================================
def test_shorter_output_losing_negation_is_rejected():
    original = (
        "Build a FastAPI backend for inventory management using PostgreSQL and SQLAlchemy. "
        "Support CRUD for products with ID, name, SKU, price, quantity. "
        "Include Pydantic schemas, validation, pagination, and error handling. "
        "Do not use a frontend. Must not add external migrations."
    )
    # Model aggressively cuts negations to save tokens
    dropped_negation_output = (
        "Build a FastAPI backend for inventory management using PostgreSQL and SQLAlchemy "
        "with CRUD, Pydantic schemas, validation, pagination, and error handling."
    )

    mock_client = _mock_groq(content=dropped_negation_output, total_tokens=180)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(original, mode="code")

    # Must be reverted to original because critical negations were lost
    assert result["decision"] == "reverted"
    assert result["enhanced_prompt"] == original
    assert "dropped explicit negation" in result["reason"]


# =========================================================================
# 7. Longer output that invents requirements (exceeds ceiling) must be rejected
# =========================================================================
def test_longer_output_exceeding_growth_ceiling_is_rejected():
    original = "Write a Python function to sort a dictionary by its values."
    
    # Model invents a massive 500-token enterprise architecture unprompted
    hallucinated_output = (
        "Create an enterprise microservice architecture with Docker, Kubernetes, AWS Lambda, "
        "Terraform CI/CD pipelines, OAuth2 authentication, Redis caching, Prometheus metrics, "
        "GraphQL gateway, Elasticsearch indexing, and write a Python function to sort a dictionary by value: "
        + " ".join(["extra invented requirement"] * 60)
    )

    mock_client = _mock_groq(content=hallucinated_output, total_tokens=450)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(original, mode="code", bypass_gate=True)

    assert result["decision"] == "reverted"
    assert result["enhanced_prompt"] == original
    assert "likely invented" in result["reason"]


# =========================================================================
# 8. Longer output that preserves intent and materially improves clarity is accepted
# =========================================================================
def test_longer_output_preserving_intent_is_accepted():
    original = "Scrape product prices from an e-commerce site with rate limiting and save to SQLite."
    structured = (
        "Develop a Python web scraper to extract product prices from an e-commerce website with the following requirements:\n"
        "- Target Fields: Extract product name, current price, and URL.\n"
        "- Rate Limiting: Implement respectful request throttling and backoff.\n"
        "- Storage: Store the extracted data into a local SQLite database table with proper schema."
    )

    mock_client = _mock_groq(content=structured, total_tokens=190)
    enhancer = PromptEnhancer(client=mock_client)
    evaluator = PromptEvaluator(client=mock_client)

    result = enhancer.enhance(original, mode="detailed")

    assert result["decision"] == "enhanced"
    assert result["enhanced_prompt"] == structured
    assert result["enhanced_tokens"] > result["original_tokens"]

    eval_res = evaluator.evaluate(original, result["enhanced_prompt"], result["enhancement_overhead_tokens"])
    assert eval_res["constraints_likely_preserved"] is True
    assert eval_res["estimated_effectiveness"] == "High (Quality & Structure Enhanced)"


# =========================================================================
# 9. Prompt with no meaningful improvement returns original
# =========================================================================
def test_no_meaningful_improvement_returns_original():
    prompt = "Calculate the Fibonacci sequence up to n terms."

    mock_client = _mock_groq(content=prompt, total_tokens=60)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(prompt, mode="code")

    assert result["decision"] == "unchanged"
    assert result["enhanced_prompt"] == prompt


# =========================================================================
# 10. Token statistics recorded accurately across all token dynamics
# =========================================================================
def test_token_accounting_on_growth_and_savings():
    evaluator = PromptEvaluator()

    # Case A: Positive token savings
    res_pos = evaluator.evaluate(
        original="Hello, please kindly write a script to reverse a string.",
        enhanced="Write a script to reverse a string.",
        enhancement_overhead_tokens=100
    )
    assert res_pos["tokens_saved"] > 0
    assert res_pos["percent_saved"] > 0
    assert res_pos["enhancement_overhead_tokens"] == 100

    # Case B: Negative token savings (longer enhanced prompt)
    res_neg = evaluator.evaluate(
        original="Build a task manager.",
        enhanced="Build a full-featured task manager with categories, priorities, due dates, and completion status.",
        enhancement_overhead_tokens=120
    )
    assert res_neg["tokens_saved"] < 0
    assert res_neg["percent_saved"] < 0
    assert res_neg["enhancement_overhead_tokens"] == 120
    assert res_neg["total_overhead_tokens"] == 120
    assert res_neg["net_positive"] is False
    assert res_neg["estimated_effectiveness"] == "High (Quality & Structure Enhanced)"


# =========================================================================
# 11. Test F: Already structured complex prompt remains unchanged (TEST F & G)
# =========================================================================
def test_already_structured_complex_prompt_remains_unchanged():
    structured_prompt = (
        "Develop a task management API using FastAPI and PostgreSQL.\n\n"
        "Functional Requirements:\n"
        "1. Task Creation: Allow creating tasks with title, description, and status.\n"
        "2. Priority Assignment: Assign priority (high, medium, low).\n"
        "3. Due Dates: Set optional due dates.\n"
        "4. Completion Status: Toggle task completion.\n"
        "5. Task Retrieval: Filter and paginate task lists.\n"
        "6. Authentication: Secure endpoints with JWT auth.\n"
        "7. Validation: Validate payloads with Pydantic.\n"
        "8. Error Handling: Return standard error responses.\n\n"
        "Deliverable:\n"
        "Provide the backend architecture and a development plan."
    )

    # 1. OptimizationGate identifies it as already structured
    gate_res = OptimizationGate.evaluate(structured_prompt)
    assert gate_res["should_optimize"] is False
    assert "already clearly structured" in gate_res["reason"]

    # 2. Even with candidate rewrite (81 -> 115 tokens) via bypass_gate=True,
    # enhancer rejects cosmetic expansion on already structured prompts
    candidate_rewrite = (
        structured_prompt + "\n\nAdditional Notes:\n- Follow clean architecture principles.\n"
        "- Ensure robust typing and modular organization across routers and database models."
    )
    mock_client = _mock_groq(content=candidate_rewrite, total_tokens=180)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(structured_prompt, mode="detailed", bypass_gate=True)

    assert result["decision"] == "reverted"
    assert result["enhanced_prompt"] == structured_prompt
    assert "already clearly structured" in result["reason"]


# =========================================================================
# 12. Test H: Messy complex prompt is meaningfully restructured
# =========================================================================
def test_messy_complex_prompt_is_meaningfully_restructured():
    messy_prompt = (
        "I want to build an employee management backend where employees can create profiles, "
        "managers can assign tasks, departments can track budgets, and administrators can view audit logs. "
        "Make sure to use Node.js, Express, and MongoDB, with role-based access control and input validation. "
        "Provide a system architecture and API specification."
    )

    # Not already structured
    assert OptimizationGate.is_already_structured(messy_prompt) is False
    gate_res = OptimizationGate.evaluate(messy_prompt)
    assert gate_res["should_optimize"] is True

    structured_output = (
        "Design an Employee Management backend using Node.js, Express, and MongoDB.\n\n"
        "Functional Requirements:\n"
        "1. Employee Profiles: Allow employees to create and manage profiles.\n"
        "2. Task Assignment: Enable managers to assign and track tasks.\n"
        "3. Department Budgets: Allow tracking department budgets.\n"
        "4. Audit Logs: Provide administrators with system audit logs.\n\n"
        "Technical Constraints:\n"
        "- Role-Based Access Control (RBAC)\n"
        "- Strict input validation\n\n"
        "Deliverables:\n"
        "- System architecture overview\n"
        "- Detailed API specification"
    )

    mock_client = _mock_groq(content=structured_output, total_tokens=220)
    enhancer = PromptEnhancer(client=mock_client)

    result = enhancer.enhance(messy_prompt, mode="detailed")

    assert result["success"] is True
    assert result["decision"] == "enhanced"
    assert result["enhanced_prompt"] == structured_output
    assert result["enhanced_tokens"] > result["original_tokens"]

