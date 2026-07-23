from inference import generate_recipe

test_prompts = [

    "Healthy breakfast for weight loss",

    "High protein vegetarian lunch",

    "Quick chicken dinner",

    "Chocolate dessert",

    "Easy Indian curry"

]

print("\n" + "=" * 60)
print("      Recipe Generator Evaluation")
print("=" * 60)

for i, prompt in enumerate(test_prompts, start=1):

    print(f"\nTest {i}")
    print("-" * 60)

    print("Prompt:")
    print(prompt)

    print("\nGenerated Recipe:\n")

    recipe = generate_recipe(prompt)

    print(recipe)

    print("-" * 60)

print("\nEvaluation Completed Successfully.")