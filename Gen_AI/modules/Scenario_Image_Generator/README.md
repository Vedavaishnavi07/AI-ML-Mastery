# AI Scenario Image Generator

An AI-powered image generation application that transforms simple user descriptions into high-quality AI-generated artwork.

The application first enhances the user's prompt using the Groq LLM and then generates an image using the free Pollinations AI image generation API. It also stores previously generated images in a gallery for quick access.

---

## Features

- Prompt enhancement using Groq LLM
- AI image generation using Pollinations AI
- Modern Gradio interface
- Gallery of previous image generations
- Download generated images
- Completely free image generation
- Real-time prompt engineering

---

## Tech Stack

- Python
- Gradio
- Groq API
- Pollinations AI
- Pillow
- Requests
- python-dotenv

---

## Project Structure

```
Scenario_Image_Generator/
│
├── app.py
├── requirements.txt
├── README.md
└── .env
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Navigate to the project

```bash
cd Scenario_Image_Generator
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```
GROQ_API_KEY=your_groq_api_key
```

---

## Run the Application

```bash
python app.py
```

Open

```
http://127.0.0.1:7860
```

---

## Example Prompt

```
A futuristic cyberpunk city at night with neon lights reflecting on wet streets while flying cars move across the skyline.
```

---

## Workflow

1. User enters a scenario.
2. Groq enhances the prompt.
3. Enhanced prompt is sent to Pollinations AI.
4. AI image is generated.
5. Image is displayed.
6. Previous generations are stored in the gallery.

---

## Future Improvements

- Image styles
- Aspect ratio selection
- Prompt history
- Multiple image generation
- Image upscaling
- Save prompt templates

---

## Author

Veda Vaishnavi