# Open-Source LLM Fine-Tuning (Recipe Assistant)

## Overview

This project demonstrates fine-tuning an open-source language model on a custom cooking recipes dataset using the Hugging Face Transformers library.

The model learns to generate simple cooking recipes based on user prompts and can be accessed through a Gradio web interface.

---

## Features

- Fine-tune an Open-Source Language Model
- Custom Recipe Dataset
- Hugging Face Trainer
- Interactive Gradio Interface
- Model Evaluation
- Local Inference
- Clean Project Structure

---

## Project Structure

```
Fine_Tuning/
│
├── app.py
├── train.py
├── inference.py
├── evaluate.py
├── README.md
├── requirements.txt
│
├── data/
│   └── recipes.json
│
├── models/
│
└── outputs/
```

---

## Dataset

The project uses a small custom JSON dataset containing cooking recipes.

Example:

```json
{
  "instruction": "How do I make pasta?",
  "output": "Boil pasta, prepare tomato sauce, mix together and serve."
}
```

---

## Training

Run

```bash
python train.py
```

This will:

- Load DistilGPT-2
- Fine-tune the model
- Save the trained model inside the `models/` folder

---

## Inference

Run

```bash
python inference.py
```

Example prompt

```
Healthy breakfast for weight loss
```

---

## Evaluation

Run

```bash
python evaluate.py
```

The script automatically tests the model using multiple cooking prompts.

---

## Launch Application

```bash
python app.py
```

Then open

```
http://127.0.0.1:7860
```

---

## Technologies Used

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Gradio

---

## Future Improvements

- Train on larger cooking datasets
- Fine-tune larger models such as Llama or Mistral using GPU
- Recipe image generation
- Nutrition estimation
- Meal planning
- Recipe recommendation system

---

## Author

Vaishnavi Penumatcha