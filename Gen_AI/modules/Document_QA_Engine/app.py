import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

PDF_PATH = "documents/MachineLearning_notes.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

docs = splitter.split_documents(documents)

pdf_chunks = [doc.page_content for doc in docs]


def retrieve_chunks(question, k=4):

    words = question.lower().split()

    scores = []

    for chunk in pdf_chunks:

        text = chunk.lower()

        score = 0

        for word in words:
            if word in text:
                score += 1

        scores.append((score, chunk))

    scores.sort(reverse=True)

    return [chunk for score, chunk in scores[:k]]


def ask_groq(prompt):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert Document Intelligence Assistant.

Always answer ONLY using the supplied document.

If the information is unavailable,
clearly say:

'I could not find this information in the uploaded document.'

Keep answers well formatted and easy to read.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def summarize_document():

    context = "\n\n".join(pdf_chunks[:12])

    prompt = f"""
Document:

{context}

Summarize this document.

Include:

- Overview
- Main Topics
- Important Concepts
- Final Summary
"""

    return ask_groq(prompt)


def key_points():

    context = "\n\n".join(pdf_chunks[:12])

    prompt = f"""
Document:

{context}

Extract the 10 most important key points.

Return them as bullet points.
"""

    return ask_groq(prompt)


def explain_simple():

    context = "\n\n".join(pdf_chunks[:10])

    prompt = f"""
Document:

{context}

Explain this document as if teaching a complete beginner.

Use very simple language.
"""

    return ask_groq(prompt)


def generate_quiz():

    context = "\n\n".join(pdf_chunks[:12])

    prompt = f"""
Document:

{context}

Create 10 multiple-choice questions.

Format:

Question

A)

B)

C)

D)

Correct Answer:
"""

    return ask_groq(prompt)


def ask_document(question):

    docs = retrieve_chunks(question)

    context = "\n\n".join(docs)

    prompt = f"""
Document Context:

{context}

Question:

{question}

Answer ONLY from the document.

If unavailable, reply:

I could not find this information in the uploaded document.
"""

    answer = ask_groq(prompt)

    source = ""

    for chunk in docs:

        source += "\n\n----------------------------\n\n"

        source += chunk[:500]

    return answer, source

with gr.Blocks(title="Document QA Engine") as demo:

    gr.Markdown(
        """
# 📄 AI Document QA Engine

Upload your knowledge document and let AI analyze it.

### Features

- 📄 Document Summary
- 🔑 Key Points
- 🎓 Explain Simply
- 📝 Generate Quiz
- ❓ Ask Questions
"""
    )

    with gr.Row():

        with gr.Column():

            question = gr.Textbox(
                label="Ask a Question",
                placeholder="Example: What is supervised learning?"
            )

            ask_btn = gr.Button("❓ Ask Document")

            summary_btn = gr.Button("📄 Summarize")

            key_btn = gr.Button("🔑 Key Points")

            explain_btn = gr.Button("🎓 Explain Simply")

            quiz_btn = gr.Button("📝 Generate Quiz")

        with gr.Column():

            output_box = gr.Textbox(
                label="AI Response",
                lines=20
            )

            source_box = gr.Textbox(
                label="Retrieved Context",
                lines=10
            )

    summary_btn.click(
        fn=lambda: (summarize_document(), ""),
        outputs=[output_box, source_box]
    )

    key_btn.click(
        fn=lambda: (key_points(), ""),
        outputs=[output_box, source_box]
    )

    explain_btn.click(
        fn=lambda: (explain_simple(), ""),
        outputs=[output_box, source_box]
    )

    quiz_btn.click(
        fn=lambda: (generate_quiz(), ""),
        outputs=[output_box, source_box]
    )

    ask_btn.click(
        fn=ask_document,
        inputs=question,
        outputs=[output_box, source_box]
    )

    question.submit(
        fn=ask_document,
        inputs=question,
        outputs=[output_box, source_box]
    )

demo.launch()