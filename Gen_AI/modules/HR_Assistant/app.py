import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

PDF_PATH = "documents/hr_policy.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

pdf_chunks = []

for page in documents:

    text = page.page_content

    chunk_size = 700
    overlap = 150

    start = 0

    while start < len(text):

        end = start + chunk_size

        pdf_chunks.append(text[start:end])

        start += chunk_size - overlap


SYSTEM_PROMPT = """
You are an AI HR Assistant.

Answer ONLY using the HR Policy provided.

If the answer cannot be found,
reply exactly:

I could not find this information in the HR Policy.

Keep answers short,
professional,
and easy to understand.

At the end always mention:

Source: HR Policy Document.
"""


def retrieve_chunks(question, top_k=3):

    words = question.lower().split()

    scored = []

    for chunk in pdf_chunks:

        score = 0

        text = chunk.lower()

        for word in words:

            if word in text:
                score += 1

        scored.append((score, chunk))

    scored.sort(reverse=True)

    return [chunk for score, chunk in scored[:top_k]]


def ask_hr(question):

    if not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    docs = retrieve_chunks(question)

    context = "\n\n".join(docs)

    prompt = f"""
HR Policy

{context}

Employee Question:

{question}

Answer ONLY using the HR Policy.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        answer = response.choices[0].message.content

    except Exception as e:

        answer = f"Error:\n\n{str(e)}"

    source = ""

    for chunk in docs:

        source += "\n"
        source += "=" * 60
        source += "\n\n"
        source += chunk

    return answer, source

with gr.Blocks(
    title="AI HR Assistant"
) as demo:

    gr.Markdown(
        """
# 🏢 AI HR Assistant (RAG)

Ask questions about the company's HR Policy.

### Features
- 📄 PDF-based Knowledge
- 🔍 Keyword Retrieval
- 🤖 Groq Llama 3.3 70B
- 📚 Source Context Display
"""
    )

    with gr.Row():

        question = gr.Textbox(
            label="Ask your HR Question",
            placeholder="Example: Can I work from home?",
            lines=2
        )

    ask_btn = gr.Button(
        "🔍 Ask HR",
        variant="primary"
    )

    answer_box = gr.Textbox(
    label="AI Answer",
    lines=10
    )

    source_box = gr.Textbox(
        label="Retrieved HR Policy Context",
        lines=14,
    )

    examples = gr.Examples(
        examples=[
            ["Can I work from home?"],
            ["How many casual leaves are allowed?"],
            ["What is the maternity leave policy?"],
            ["What happens if I resign?"],
            ["What is the notice period?"],
            ["What are office working hours?"]
        ],
        inputs=question
    )

    ask_btn.click(
        fn=ask_hr,
        inputs=question,
        outputs=[
            answer_box,
            source_box
        ]
    )

    question.submit(
        fn=ask_hr,
        inputs=question,
        outputs=[
            answer_box,
            source_box
        ]
    )

demo.launch()