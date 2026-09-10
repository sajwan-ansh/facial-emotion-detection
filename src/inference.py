import numpy as np
import tensorflow as tf

from config import EMOTIONS
from utils import preprocess_face


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_face(model, face):
    probabilities = model.predict(preprocess_face(face), verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities /= probabilities.sum()
    return probabilities


def prediction_from_probabilities(probabilities, min_confidence=0.45):
    probabilities = np.asarray(probabilities)
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = EMOTIONS[index] if confidence >= min_confidence else "Uncertain"
    return label, confidence, index
