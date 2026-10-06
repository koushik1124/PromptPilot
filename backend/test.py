import logging
logging.basicConfig(level=logging.INFO)

from app.services.optimization_gate import OptimizationGate, is_already_structured
from app.services.prompt_enhancer import PromptEnhancer

p = (
    "Write a Python program that:\n"
    "1. Reads a CSV file.\n"
    "2. Finds and removes duplicate rows.\n"
    "3. Saves the result as 'cleaned.csv'.\n"
    "4. Handles the case where the input file does not exist.\n"
    "5. Keep the implementation simple."
)

print("is_already_structured:", is_already_structured(p))   # expect True
print("gate:", OptimizationGate.evaluate(p, mode=None)["reason"])  # expect structured-block

e = PromptEnhancer()
for m in ("auto", "code"):
    r = e.enhance(p, mode=m)
    print(f"mode={m}: {r['decision']} via {r['model']} "
          f"({r['original_tokens']} -> {r['enhanced_tokens']})")