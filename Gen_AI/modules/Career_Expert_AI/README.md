# 💼 Career Expert AI

An AI-powered career guidance assistant built using Groq's Llama 3.3 model. It helps users with resume advice, interview preparation, career planning, skill recommendations, and job guidance through a simple Gradio interface.

## Features

- Career guidance
- Resume improvement suggestions
- Interview preparation
- Skill recommendations
- Job role exploration
- Conversation history
- Fast responses using Groq API

## Technologies

- Python
- Groq API
- OpenAI Python SDK
- Gradio
- python-dotenv

## Project Structure

```
Career_Expert_AI/
│
├── app.py
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

- Voice interaction
- Resume upload
- PDF export
- Job matching