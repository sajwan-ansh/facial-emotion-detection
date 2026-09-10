import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import MODEL_PATH, NUM_CLASSES, OUTPUT_DIR
from model import build_model
from utils import load_fer2013


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), _ = load_fer2013()
    if args.sample_size:
        n = min(args.sample_size, len(x_train))
        x_train, y_train = x_train[:n], y_train[:n]

    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=12, width_shift_range=0.10, height_shift_range=0.10,
        zoom_range=0.12, shear_range=0.08, horizontal_flip=True, fill_mode="reflect"
    )
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    history = model.fit(
        augmentation.flow(x_train, y_train_oh, batch_size=args.batch_size),
        validation_data=(x_val, y_val_oh), epochs=args.epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150); plt.close(fig)
    print(f"Best model saved to {args.model_path}")
    print("Class weights:", class_weights)


if __name__ == "__main__":
    main()
