import gradio as gr
from inference import generate_recipe


def generate(recipe_request):

    answer = generate_recipe(recipe_request)

    return answer


with gr.Blocks(title="AI Recipe Generator") as demo:

    gr.Markdown(
        """
# 🍳 AI Recipe Generator

Fine-Tuned DistilGPT2 using Hugging Face Trainer

Enter any cooking request below.
"""
    )

    prompt = gr.Textbox(
        label="Recipe Request",
        placeholder="Example: Healthy Pasta Recipe"
    )

    output = gr.Textbox(
        label="Generated Recipe",
        lines=15
    )

    button = gr.Button("Generate Recipe")

    button.click(
        generate,
        inputs=prompt,
        outputs=output
    )

demo.launch()