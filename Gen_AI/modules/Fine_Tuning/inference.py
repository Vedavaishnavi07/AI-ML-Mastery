from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "models"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def generate_recipe(prompt):

    text = f"Instruction:\n{prompt}\n\nRecipe:\n"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "Recipe:" in result:
        result = result.split("Recipe:", 1)[1].strip()

    return result


if __name__ == "__main__":

    question = input("Enter Recipe Request: ")

    print("\n")

    print(generate_recipe(question))