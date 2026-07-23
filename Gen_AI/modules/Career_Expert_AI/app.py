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

messages = [
    {
        "role": "system",
        "content": """
You are CareerMentor AI.

You help students and professionals with:

- Resume Review
- Interview Preparation
- Career Guidance
- Skill Roadmaps
- AI & Data Science
- Python
- SQL
- Machine Learning
- Deep Learning
- GenAI
- Salary Guidance
- Higher Studies

Always answer professionally.

Use bullet points whenever possible.

End every answer with one practical next step.
"""
    }
]


def chat(user_message, history):

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5
    )

    answer = response.choices[0].message.content

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    history.append(
        (
            user_message,
            answer
        )
    )

    return history, history


def clear_chat():

    global messages

    messages = [messages[0]]

    return [], []


with gr.Blocks(title="Career Mentor AI") as demo:

    gr.Markdown(
        """
# 🎯 Career Mentor AI

Your personal AI Career Coach
"""
    )

    chatbot = gr.Chatbot(height=500)

    state = gr.State([])

    msg = gr.Textbox(
        placeholder="Ask anything about your career..."
    )

    with gr.Row():

        send = gr.Button("🚀 Send")

        clear = gr.Button("🗑 Clear")

    send.click(
        chat,
        inputs=[msg, state],
        outputs=[chatbot, state]
    )

    msg.submit(
        chat,
        inputs=[msg, state],
        outputs=[chatbot, state]
    )

    clear.click(
        clear_chat,
        outputs=[chatbot, state]
    )

demo.launch()