# 📄 AI Document QA Engine

An AI-powered document intelligence application that allows users to analyze PDF documents using Groq's Llama 3.3 model.

The application can summarize documents, extract key points, explain concepts in simple language, generate quizzes, and answer questions based on the uploaded document.

## Features

- PDF document analysis
- AI-generated summaries
- Key point extraction
- Explain Like I'm Five
- Quiz generation
- Document question answering
- Retrieved context display
- Keyword-based retrieval
- Groq Llama 3.3 integration

## Technologies

- Python
- Groq API
- LangChain
- PyPDF
- Gradio

## Project Structure

```
Document_QA_Engine/
│
├── app.py
├── documents/
│   └── MachineLearning_notes.pdf
├── README.md
├── requirements.txt
└── outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_api_key
```

Run:

```bash
python app.py
```

## Available Features

- Document Summary
- Key Points
- Explain Simply
- Generate Quiz
- Ask Questions
- Retrieved Context

## Model

- Llama-3.3-70B-Versatile (Groq)

## Future Improvements

- Multiple document upload
- Semantic search
- ChromaDB integration
- Conversation history
- OCR support