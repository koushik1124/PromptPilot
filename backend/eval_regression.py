"""
Regression eval harness for PromptPilot.

Runs a fixed set of prompts across all four modes plus auto, records
enhancer + evaluator output, and diffs against a saved baseline. This
is the concrete replacement for asking "how long will this wording
hold" — that question has no defensible numeric answer; this gives
you a checkable signal instead.

Costs real Groq tokens every run — this is not free to execute.

Usage:
    python eval_regression.py --save-baseline   # first run: creates eval_baseline.json
    python eval_regression.py                   # subsequent runs: diffs against baseline
    python eval_regression.py --deep-verify      # also runs the LLM judge (extra Groq cost)
"""

import json
import argparse
from pathlib import Path

from app.services.prompt_enhancer import PromptEnhancer
from app.services.prompt_evaluator import PromptEvaluator

BASELINE_PATH = Path("eval_baseline.json")

# Deliberately spans: already-optimal, underspecified, dense-constraint,
# and short/simple cases per mode, since these are the shapes most
# likely to expose drift in either direction (over-trimming vs.
# under-trimming, or misclassification in auto mode).
TEST_CASES = [
    {"name": "concise_already_short", "mode": "concise",
     "prompt": "List the planets in order from the sun."},
    {"name": "concise_with_filler", "mode": "concise",
     "prompt": "Could you please help me write a function that reverses a string?"},

    {"name": "detailed_underspecified", "mode": "detailed",
     "prompt": "Build a login system."},
    {"name": "detailed_already_specific", "mode": "detailed",
     "prompt": "Build a FastAPI login endpoint using JWT tokens, bcrypt password hashing, "
               "and rate limiting on failed attempts."},

    {"name": "code_dense_constraints", "mode": "code",
     "prompt": "Write a Python function using pandas that reads a CSV, drops rows with null "
               "values in the 'email' column, and returns a deduplicated DataFrame."},
    {"name": "code_simple", "mode": "code",
     "prompt": "Write a function to check if a number is prime."},

    {"name": "creative_tone_sensitive", "mode": "creative",
     "prompt": "Write a short, melancholic poem about autumn leaves falling."},
    {"name": "creative_brand_voice", "mode": "creative",
     "prompt": "Write a punchy, confident tagline for a sneaker brand aimed at Gen Z runners."},

    {"name": "auto_code_signal", "mode": "auto",
     "prompt": "Explain how async/await works in JavaScript with a short example."},
    {"name": "auto_creative_signal", "mode": "auto",
     "prompt": "Write a two-sentence story about a lighthouse keeper who never sees the ocean."},
    {"name": "auto_ambiguous", "mode": "auto",
     "prompt": "Summarize this document in three bullet points."},
]

# Fields that matter for detecting drift. Token counts and raw "reason"
# text are expected to vary slightly run to run at temperature=0.1 —
# these fields shouldn't, and a change here is the actual signal.
TRACKED_FIELDS = (
    "decision",
    "detected_mode",
    "constraint_coverage_ratio",
    "negations_preserved",
    "estimated_effectiveness",
)


def run_case(enhancer, evaluator, case, deep_verify=False):
    result = enhancer.enhance(case["prompt"], mode=case["mode"])

    eval_result = None
    if result.get("success"):
        eval_result = evaluator.evaluate(
            original=result["original_prompt"].strip(),
            enhanced=result["enhanced_prompt"],
            enhancement_overhead_tokens=result.get("enhancement_overhead_tokens") or 0,
            deep_verify=deep_verify,
        )

    return {
        "name": case["name"],
        "requested_mode": case["mode"],
        "decision": result.get("decision"),
        "detected_mode": result.get("detected_mode"),
        "original_tokens": result.get("original_tokens"),
        "enhanced_tokens": result.get("enhanced_tokens"),
        "tokens_saved": result.get("tokens_saved"),
        "constraint_coverage_ratio": (
            eval_result.get("constraint_coverage_ratio") if eval_result else None
        ),
        "negations_preserved": (
            eval_result.get("negations_preserved") if eval_result else None
        ),
        "estimated_effectiveness": (
            eval_result.get("estimated_effectiveness") if eval_result else None
        ),
        "reason": result.get("reason"),
    }


def run_all(deep_verify=False):
    enhancer = PromptEnhancer()
    evaluator = PromptEvaluator()
    return [run_case(enhancer, evaluator, case, deep_verify=deep_verify) for case in TEST_CASES]


def save_baseline(results):
    BASELINE_PATH.write_text(json.dumps(results, indent=2))
    print(f"Baseline saved to {BASELINE_PATH} ({len(results)} cases).")


def load_baseline():
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def diff_results(baseline, current):
    baseline_by_name = {r["name"]: r for r in baseline}
    changes = []

    for cur in current:
        base = baseline_by_name.get(cur["name"])
        if base is None:
            changes.append((cur["name"], "NEW_CASE", None, cur))
            continue

        for field in TRACKED_FIELDS:
            if base.get(field) != cur.get(field):
                changes.append((cur["name"], field, base.get(field), cur.get(field)))

    missing = set(baseline_by_name) - {r["name"] for r in current}
    for name in missing:
        changes.append((name, "MISSING_CASE", baseline_by_name[name], None))

    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true",
                         help="Overwrite the baseline with this run's results.")
    parser.add_argument("--deep-verify", action="store_true",
                         help="Also run the LLM judge verification (extra Groq cost per case).")
    args = parser.parse_args()

    results = run_all(deep_verify=args.deep_verify)

    if args.save_baseline or not BASELINE_PATH.exists():
        save_baseline(results)
        return

    baseline = load_baseline()
    changes = diff_results(baseline, results)

    if not changes:
        print(f"No drift detected across {len(results)} cases.")
        return

    print(f"DRIFT DETECTED — {len(changes)} field(s) changed vs baseline:\n")
    for name, field, old, new in changes:
        print(f"  [{name}] {field}: {old!r} -> {new!r}")


if __name__ == "__main__":
    main()