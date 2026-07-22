import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "../model/final_model.keras"
CLASS_NAMES_PATH = "../model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1]

    result = "🏆 Top Predictions\n\n"

    for i in top_indices[:5]:
        result += f"{class_names[i].title()} : {predictions[i] * 100:.2f}%\n"

    return result


examples = [
    "../data/dataset/birds",
    "../data/dataset/cars",
    "../data/dataset/cats",
    "../data/dataset/dogs",
    "../data/dataset/flowers"
]


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="🖼️ Multi-Class Image Classifier",
    description="""
Upload an image and the model will classify it into one of the following classes:

🐦 Birds
🚗 Cars
🐱 Cats
🐶 Dogs
🌸 Flowers

Built using TensorFlow, ResNet50 Transfer Learning and Gradio.
"""
)

interface.launch()