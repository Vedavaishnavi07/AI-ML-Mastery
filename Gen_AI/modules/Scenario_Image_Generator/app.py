import os
import urllib.parse
from pathlib import Path

import gradio as gr
import requests

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

gallery = []


def enhance_prompt(prompt):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": """
You are an expert prompt engineer.

Improve the user's prompt for AI image generation.

Always include:

• highly detailed
• cinematic lighting
• ultra realistic
• masterpiece
• vibrant colors
• 8k quality
• professional photography

Return ONLY the improved prompt.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def generate_image(prompt):

    if prompt.strip() == "":
        return "", None, gallery

    enhanced_prompt = enhance_prompt(prompt)

    encoded_prompt = urllib.parse.quote(enhanced_prompt)

    image_url = (
        f"https://image.pollinations.ai/prompt/"
        f"{encoded_prompt}"
    )

    image = Image.open(
        requests.get(
            image_url,
            stream=True
        ).raw
    )

    gallery.append(image)

    return (
        enhanced_prompt,
        image,
        gallery
    )
with gr.Blocks(
    title="Scenario Image Generator"
) as demo:

    gr.Markdown(
        """
# 🎨 AI Scenario Image Generator

Generate stunning AI images from your imagination.

### 🚀 Features
- Prompt Enhancement using Groq
- AI Image Generation using Pollinations
- Previous Images Gallery
- Download Generated Image
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            prompt_box = gr.Textbox(
                label="Describe your scenario",
                lines=5,
                placeholder="Example: A futuristic city floating above the clouds during sunset."
            )

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary"
            )

            enhanced_box = gr.Textbox(
                label="Enhanced Prompt",
                lines=8
            )

        with gr.Column(scale=2):

            image_output = gr.Image(
                label="Generated Image",
                type="pil"
            )

    gallery_output = gr.Gallery(
        label="Previous Generations",
        columns=3,
        rows=2,
        height=350
    )

    generate_btn.click(
        fn=generate_image,
        inputs=prompt_box,
        outputs=[
            enhanced_box,
            image_output,
            gallery_output
        ]
    )

    prompt_box.submit(
        fn=generate_image,
        inputs=prompt_box,
        outputs=[
            enhanced_box,
            image_output,
            gallery_output
        ]
    )

demo.launch()