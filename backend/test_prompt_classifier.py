from app.services.prompt_classifier import PromptClassifier


classifier = PromptClassifier()


TEST_PROMPTS = [

    # ================================================================
    # 1. CODE
    # ================================================================

    (
        "Good Code Prompt",
        """
Write a Python function that sorts a dictionary by its values in
descending order and include a short example.
""",
        "code",
    ),

    (
        "Good FastAPI Prompt",
        """
Build a FastAPI backend for a task management app. Use PostgreSQL and
JWT authentication. Users can create, update, delete, and search tasks.
Each user must only see their own tasks. Include proper error handling.
""",
        "code",
    ),

    (
        "Good Simple Task",
        """
Write a Python function that checks whether a string is a palindrome.
Return True or False and include two examples.
""",
        "code",
    ),

    (
        "Good SQL Task",
        """
Write a SQL query that returns the top 5 customers by total purchase
amount from an orders table.
""",
        "code",
    ),


    # ================================================================
    # 2. MESSY BUT CLEAR CODE REQUESTS
    # ================================================================

    (
        "Messy FastAPI",
        """
i want to build a fastapi backend for a task management app users
should be able to create update delete and search tasks and each user
should only see their own tasks use postgres and i want jwt authentication
and proper error handling
""",
        "code",
    ),

    (
        "Messy Python",
        """
write python code that reads a csv and removes duplicate rows and then
save it as another csv file and make sure missing values don't crash it
""",
        "code",
    ),

    (
        "Messy Frontend",
        """
make a react dashboard for users where they can see their profile
and some statistics and recent activity use material ui and make it
responsive
""",
        "code",
    ),


    # ================================================================
    # 3. AMBIGUOUS
    #
    # These do not contain enough information to reliably determine
    # a specific task mode.
    # ================================================================

    (
        "Ambiguous 1",
        "make this better",
        None,
    ),

    (
        "Ambiguous 2",
        "fix this",
        None,
    ),

    (
        "Ambiguous 3",
        "improve it",
        None,
    ),

    (
        "Ambiguous 4",
        "make it professional",
        None,
    ),


    # ================================================================
    # 4. SHORT BUT CLEAR
    # ================================================================

    (
        "Short Code",
        "Write a Python function to reverse a string.",
        "code",
    ),

    (
        "Short Explanation",
        "Explain recursion simply.",
        "detailed",
    ),

    (
        "Short SQL",
        "Find duplicate emails in a users table.",
        "code",
    ),

    (
        "Short Creative",
        "Give me 10 startup ideas for students.",
        "creative",
    ),


    # ================================================================
    # 5. CREATIVE / IDEATION
    # ================================================================

    (
        "Creative 1",
        """
Give me ideas for an AI app that helps college students manage their
studies.
""",
        "creative",
    ),

    (
        "Creative 2",
        """
Give me some unique AI startup ideas.
""",
        "creative",
    ),

    (
        "Creative 3",
        """
I want to build something with AI for college students. Give me ideas.
""",
        "creative",
    ),

    (
        "Creative 4",
        """
Write a short futuristic story about an AI assistant that becomes
self-aware.
""",
        "creative",
    ),


    # ================================================================
    # 6. VERBOSE REQUESTS
    #
    # Length should NOT change the classification.
    # These are still primarily code tasks.
    # ================================================================

    (
        "Verbose Code",
        """
I would like you to please write a Python function for me that takes
a list of numbers as its input and then performs the task of sorting
those numbers from the smallest number to the largest number and after
doing that I would also like you to provide a simple example showing
how the function works.
""",
        "code",
    ),

    (
        "Verbose Request",
        """
I want you to create a Python program for me. The main purpose of this
program should be to read a CSV file. After reading the CSV file, the
program should look through all the rows and identify rows that are
duplicates of other rows. Those duplicate rows should then be removed.
Finally, the resulting cleaned data should be saved into another CSV
file.
""",
        "code",
    ),


    # ================================================================
    # 7. TECHNICAL CONSTRAINTS
    #
    # Constraints should NOT change the primary mode.
    # ================================================================

    (
        "Technical Constraints",
        """
Build a Python API using FastAPI and PostgreSQL. Do not use MongoDB.
Do not use Flask. Use JWT authentication.
""",
        "code",
    ),

    (
        "Frontend Constraint",
        """
Build a React dashboard using Material UI. Do not use Tailwind CSS.
Keep the design responsive.
""",
        "code",
    ),


    # ================================================================
    # 8. COMPLEX BUT ALREADY STRUCTURED
    # ================================================================

    (
        "Structured Complex",
        """
Build a FastAPI task management backend.

Requirements:
1. PostgreSQL database.
2. JWT authentication.
3. Users can create, update, delete, and search tasks.
4. Users can only access their own tasks.
5. Validate request data.
6. Return appropriate HTTP errors.
7. Organize the application using routers, services, and database models.

Return the project structure and implementation.
""",
        "code",
    ),


    # ================================================================
    # 9. COMPLEX UNSTRUCTURED
    # ================================================================

    (
        "Complex Unstructured",
        """
I want to build a task management backend with FastAPI and postgres
where users login with jwt and can create update delete and search
their tasks and users shouldn't be able to see another user's tasks
and I want proper validation and error handling and a clean project
structure with routers services and models
""",
        "code",
    ),


    # ================================================================
    # 10. DETAILED / EXPLANATION
    # ================================================================

    (
        "Explain Concept",
        """
Explain how the Python garbage collector works and why memory
management is important.
""",
        "detailed",
    ),

    (
        "Technical Explanation",
        """
Explain the difference between PostgreSQL and MongoDB and when each
type of database is appropriate.
""",
        "detailed",
    ),

    (
        "Learning Request",
        """
Teach me how REST APIs work from the basics with practical examples.
""",
        "detailed",
    ),

    (
        "Comparison",
        """
Compare FastAPI, Flask, and Django in terms of architecture,
performance, and typical use cases.
""",
        "detailed",
    ),


    # ================================================================
    # 11. CONCISE
    # ================================================================

    (
        "Summarization",
        """
Summarize this article in 5 bullet points.
""",
        "concise",
    ),

    (
        "Shorten Text",
        """
Make this paragraph shorter while keeping the important information.
""",
        "concise",
    ),

    (
        "Concise Rewrite",
        """
Rewrite this email in a concise and professional way.
""",
        "concise",
    ),

    (
        "TLDR",
        """
Give me the TL;DR of this document.
""",
        "concise",
    ),


    # ================================================================
    # 12. MIXED / EDGE CASES
    # ================================================================

    (
        "Creative Technical Idea",
        """
Come up with five creative ideas for an AI-powered cybersecurity tool.
""",
        "creative",
    ),

    (
        "Code Explanation",
        """
Explain this Python code line by line and tell me how it works.
""",
        "detailed",
    ),

    (
        "Code Generation",
        """
Create a Python script that monitors a folder and moves new CSV files
to another directory.
""",
        "code",
    ),

    (
        "Concise Technical Summary",
        """
Summarize the main differences between REST and GraphQL in three
sentences.
""",
        "concise",
    ),
]


# ================================================================
# RUN TESTS
# ================================================================

passed = 0
failed = 0
ambiguous = 0


for name, prompt, expected_mode in TEST_PROMPTS:

    result = classifier.classify(prompt)

    actual_mode = result.get("mode")

    if expected_mode is None:
        status = "AMBIGUOUS"

        if actual_mode is None:
            passed += 1
        else:
            failed += 1

        ambiguous += 1

    elif actual_mode == expected_mode:
        status = "PASS"
        passed += 1

    else:
        status = "FAIL"
        failed += 1

    print("=" * 80)
    print(name)
    print("Prompt:", prompt.strip())
    print("Expected:", expected_mode)
    print("Actual:", actual_mode)
    print("Status:", status)
    print("Full Result:", result)


# ================================================================
# SUMMARY
# ================================================================

total = len(TEST_PROMPTS)

print("\n" + "=" * 80)
print("CLASSIFIER TEST SUMMARY")
print("=" * 80)

print(f"Total Tests : {total}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")
print(f"Ambiguous   : {ambiguous}")

if total:
    accuracy = (passed / total) * 100
    print(f"Accuracy    : {accuracy:.2f}%")

print("=" * 80)