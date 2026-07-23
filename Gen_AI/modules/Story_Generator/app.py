import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

story_messages = []
current_story = ""


def story_stats(text):

    words = len(text.split())
    characters = len(text)
    lines = len(text.splitlines())

    return f"""
## 📊 Story Analytics

**Words:** {words}

**Characters:** {characters}

**Lines:** {lines}
"""


def generate_story(genre, theme, hero, setting, length):

    global story_messages
    global current_story

    story_messages = [
        {
            "role": "system",
            "content": """
You are an award-winning storyteller.

Write cinematic stories.

Create immersive worlds.

Remember every previous event.

Never restart the story.

Always end with suspense.

Keep the story engaging and creative.
"""
        }
    ]

    prompt = f"""
Create a story.

Genre: {genre}

Theme: {theme}

Hero: {hero}

Setting: {setting}

Length: {length}
"""

    story_messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=story_messages,
        temperature=0.9
    )

    answer = response.choices[0].message.content

    story_messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    current_story = answer

    return current_story, story_stats(current_story)


def continue_story(action):

    global story_messages
    global current_story

    if len(story_messages) == 0:

        return "Generate a story first.", story_stats("")

    story_messages.append(
        {
            "role": "user",
            "content": f"""
Continue the story.

Player Action:

{action}
"""
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=story_messages,
        temperature=0.9
    )

    answer = response.choices[0].message.content

    story_messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    current_story += "\n\n" + answer

    return current_story, story_stats(current_story)


def save_story():

    if current_story.strip() == "":
        return "Nothing to save."

    Path("outputs").mkdir(exist_ok=True)

    with open(
        "outputs/story.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(current_story)

    return "Story saved successfully."

with gr.Blocks(title="AI Story Studio") as demo:

    gr.Markdown("# 📖 AI Story Studio")
    gr.Markdown("### Create interactive AI-powered stories")

    with gr.Row():

        with gr.Column():

            genre = gr.Dropdown(
                ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Horror"],
                value="Fantasy",
                label="Genre"
            )

            theme = gr.Dropdown(
                ["Epic", "Dark", "Funny", "Emotional"],
                value="Epic",
                label="Theme"
            )

            hero = gr.Textbox(
                label="Hero",
                placeholder="Aria"
            )

            setting = gr.Textbox(
                label="Setting",
                placeholder="Ancient Kingdom"
            )

            length = gr.Radio(
                ["Short", "Medium", "Long"],
                value="Medium",
                label="Length"
            )

            generate_btn = gr.Button(
                "✨ Generate Story",
                variant="primary"
            )

            gr.Markdown("## ⚡ Continue Story")

            action = gr.Textbox(
                label="Custom Action",
                placeholder="Open the mysterious door..."
            )

            with gr.Row():

                fight_btn = gr.Button("⚔ Fight")

                explore_btn = gr.Button("🗺 Explore")

            with gr.Row():

                talk_btn = gr.Button("💬 Talk")

                escape_btn = gr.Button("🏃 Escape")

            continue_btn = gr.Button(
                "➡ Continue Story"
            )

            save_btn = gr.Button(
                "💾 Save Story"
            )

            save_status = gr.Textbox(
                label="Status",
                interactive=False
            )

        with gr.Column():

            story = gr.Textbox(
                label="Story",
                lines=28
            )

            analytics = gr.Markdown(
                """
## 📊 Story Analytics

No story generated.
"""
            )

    generate_btn.click(
        fn=generate_story,
        inputs=[
            genre,
            theme,
            hero,
            setting,
            length
        ],
        outputs=[
            story,
            analytics
        ]
    )

    continue_btn.click(
        fn=continue_story,
        inputs=action,
        outputs=[
            story,
            analytics
        ]
    )

    fight_btn.click(
        fn=lambda: continue_story("Fight the enemy."),
        outputs=[
            story,
            analytics
        ]
    )

    explore_btn.click(
        fn=lambda: continue_story("Explore the surroundings."),
        outputs=[
            story,
            analytics
        ]
    )

    talk_btn.click(
        fn=lambda: continue_story("Talk to the nearby character."),
        outputs=[
            story,
            analytics
        ]
    )

    escape_btn.click(
        fn=lambda: continue_story("Escape from danger."),
        outputs=[
            story,
            analytics
        ]
    )

    save_btn.click(
        fn=save_story,
        outputs=save_status
    )

if __name__ == "__main__":

    demo.launch(
        inbrowser=True
    )