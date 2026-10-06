from app.services.prompt_enhancer import PromptEnhancer

enhancer = PromptEnhancer()

test_cases = [
    {
        "name": "Simple CSV task",
        "prompt": (
            "make a python script to read csv "
            "and remove duplicates"
        ),
    },
    {
        "name": "Complex FastAPI task",
        "prompt": """
Build a FastAPI backend for an inventory management
system using PostgreSQL and SQLAlchemy.

The API should support creating, updating, deleting,
and listing products. Each product should have an ID,
name, SKU, quantity, and price.

Add request validation, pagination for listing products,
and proper error handling for missing products and
duplicate SKUs.

Use Pydantic schemas and organize the code into routers,
services, and database models.

Include the database configuration and the API endpoints.
Do not use a frontend.
""",
    },
    {
        "name": "Existing code modification",
        "prompt": """
I have an existing FastAPI application with a PostgreSQL
database and SQLAlchemy models.

Add an endpoint that returns the total number of active
users and the number of users created in the last 30 days.

Use the existing database session and user model.
Do not modify the existing schema or add new dependencies.

Return the endpoint code and explain where it belongs.
""",
    },
]

for case in test_cases:
    print("\n" + "=" * 70)
    print("TEST:", case["name"])
    print("=" * 70)

    result = enhancer.enhance(
        prompt=case["prompt"],
        mode="code",
    )

    print("Decision:", result.get("decision"))
    print("Original:", result["original_prompt"])
    print("Enhanced:", result["enhanced_prompt"])
    print("Reason:", result.get("reason"))
    print("Overhead tokens:", result.get("enhancement_overhead_tokens"))
    print("Success:", result.get("success"))
    print("Message:", result.get("message"))
