from app.services.prompt_enhancer import PromptEnhancer
from app.services.prompt_evaluator import PromptEvaluator

enhancer = PromptEnhancer()
evaluator = PromptEvaluator()

prompts = [
    {
        "name": "Simple CSV",
        "prompt": "make a python script to read csv and remove duplicates",
    },
    {
        "name": "Complex API",
        "prompt": """
Build a FastAPI backend for inventory management using
PostgreSQL and SQLAlchemy.

Support creating, updating, deleting, and listing products.
Each product has an ID, name, SKU, quantity, and price.

Use Pydantic validation, pagination, and error handling
for missing products and duplicate SKUs.

Organize code into routers, services, and database models.
Include database configuration and API endpoints.
Do not use a frontend.
""",
    },
    {
        "name": "Existing code constraints",
        "prompt": """
Add an endpoint to my existing FastAPI app that returns
total active users and users created in the last 30 days.

Use the existing PostgreSQL session and SQLAlchemy user
model. Do not modify the schema or add dependencies.

Return the endpoint code and explain where it belongs.
""",
    },
]

for item in prompts:
    print("\n" + "=" * 60)
    print("TEST:", item["name"])
    print("=" * 60)

    result = enhancer.enhance(
        prompt=item["prompt"],
        mode="code",
    )

    print("Decision:", result.get("decision"))
    print("Success:", result.get("success"))
    print("Enhanced:", result["enhanced_prompt"])
    print("Enhancer overhead tokens:", result.get("enhancement_overhead_tokens"))

    if not result.get("success"):
        print("Skipping evaluation — enhancement failed:", result.get("message"))
        continue

    eval_result = evaluator.evaluate(
        original=item["prompt"].strip(),
        enhanced=result["enhanced_prompt"],
        enhancement_overhead_tokens=result.get("enhancement_overhead_tokens", 0),
        deep_verify=False,  # flip to True only for manual spot-checks — costs an extra Groq call
    )

    print("\n--- Evaluation ---")
    print("Tokens saved (raw):", eval_result["tokens_saved"], f"({eval_result['percent_saved']}%)")
    print("Net tokens saved (after overhead):", eval_result["net_tokens_saved"])
    print("Net positive:", eval_result["net_positive"])
    print("Clarity delta:", eval_result["clarity_delta"])
    print("Constraint coverage ratio:", eval_result["constraint_coverage_ratio"])
    print("Constraints likely preserved:", eval_result["constraints_likely_preserved"])
    if eval_result["constraint_terms_dropped"]:
        print("  Dropped terms:", eval_result["constraint_terms_dropped"])
    if eval_result["possible_additions"]:
        print("  Possible additions (review):", eval_result["possible_additions"])
    print("Effectiveness:", eval_result["estimated_effectiveness"])