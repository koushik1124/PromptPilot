from app.services.prompt_enhancer import PromptEnhancer

enhancer = PromptEnhancer()

result = enhancer.enhance(
    prompt="make a python script to read csv and remove duplicates",
    mode="code",
)

print("Success:", result.get("success"))
print("Original:", result["original_prompt"])
print("Enhanced:", result["enhanced_prompt"])
print("Decision:", result.get("decision"))
print("Reason:", result.get("reason"))
print("Overhead tokens:", result.get("enhancement_overhead_tokens"))
print("Message:", result.get("message"))
