import numpy as np
import pandas as pd

from config import DATA_PATH, EMOTIONS, IMG_SIZE, NUM_CLASSES


def load_fer2013(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels", "Usage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FER-2013 CSV is missing columns: {sorted(missing)}")
    splits = {}
    for usage, split_name in {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}.items():
        subset = df[df["Usage"] == usage]
        if subset.empty:
            raise ValueError(f"FER-2013 split '{usage}' is empty")
        pixels = np.stack([np.fromstring(value, sep=" ", dtype=np.float32) for value in subset["pixels"]])
        if pixels.shape[1] != IMG_SIZE * IMG_SIZE:
            raise ValueError("Unexpected FER-2013 image size")
        images = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        labels = subset["emotion"].to_numpy(dtype=np.int64)
        if np.any((labels < 0) | (labels >= NUM_CLASSES)):
            raise ValueError("Emotion labels are outside the supported 0-6 range")
        splits[split_name] = (images, labels)
    return splits["train"], splits["validation"], splits["test"]


def preprocess_face(face):
    face = np.asarray(face)
    if face.ndim != 2:
        raise ValueError(f"Expected a grayscale 2-D face crop, got shape {face.shape}")
    face = np.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    if face.max() > 1.0:
        face /= 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def emotion_name(index):
    return EMOTIONS[int(index)]
