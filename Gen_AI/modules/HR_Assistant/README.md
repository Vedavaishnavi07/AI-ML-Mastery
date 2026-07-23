# 🏢 AI HR Assistant

An enterprise HR assistant that answers employee questions using an HR policy document. The application retrieves relevant sections from the HR PDF and uses Groq's Llama 3.3 model to generate grounded responses.

## Features

- HR policy question answering
- Retrieval-Augmented Generation (RAG)
- PDF document parsing
- Source context display
- Groq Llama 3.3 integration
- Gradio interface

## Technologies

- Python
- Groq API
- LangChain
- PyPDF
- Gradio

## Project Structure

```
HR_Assistant/
│
├── app.py
├── documents/
│   └── hr_policy.pdf
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

## Model

- Llama-3.3-70B-Versatile (Groq)

## Future Improvements

- Semantic embeddings
- FAISS vector database
- Multiple HR documents
- Employee authentication