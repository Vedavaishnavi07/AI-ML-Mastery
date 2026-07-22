# Multi-Class Image Classifier using ResNet50

##  Project Overview

This project is a Deep Learning based Multi-Class Image Classification application built using TensorFlow and Transfer Learning with ResNet50.

The model classifies images into five categories:

- Birds
- Cars
- Cats
- Dogs
- Flowers

A Gradio web interface is included for real-time image prediction.

---

## Features

- Transfer Learning using ResNet50
- Data Augmentation
- Fine-tuning of pretrained layers
- Early Stopping
- Model Checkpointing
- Automatic Class Name Saving
- Gradio Web Application
- Accuracy & Loss Visualization

---

## Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- Gradio
- Matplotlib
- NumPy

---

## Project Structure

```
Image_Classifier/
│
├── app/
│   └── app.py
│
├── data/
│   └── dataset/
│
├── model/
│   ├── image_classifier.keras
│   ├── final_model.keras
│   └── class_names.json
│
├── outputs/
│   ├── accuracy.png
│   └── loss.png
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Model Architecture

- Base Model: ResNet50 (ImageNet Weights)
- Global Average Pooling
- Dense Layer (256 Neurons)
- Dropout (0.5)
- Output Layer (Softmax)

---

## Training Strategy

- Image Size: 224 × 224
- Batch Size: 32
- Validation Split: 20%
- Optimizer: Adam
- Learning Rate: 1e-5
- Early Stopping
- Model Checkpoint
- Fine-tuning of last ResNet50 layers

---

## Results

The model was trained using transfer learning and evaluated on a validation dataset.

Training graphs are automatically generated:

- Accuracy Curve
- Loss Curve

---

## Gradio Interface

The project includes a Gradio application that allows users to upload an image and receive predictions.

Example:

```
Prediction

Dogs

Confidence: 92%
```

---

## How to Run

### Clone Repository

```bash
git clone <repository-url>
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Environment

Windows

```bash
myenv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Application

```bash
python app.py
```

---

## Future Improvements

- More image categories
- Larger dataset
- Hugging Face deployment
- Mobile-friendly interface
- Model optimization for faster inference

---

## Author

Veda Vaishnavi Penumatcha