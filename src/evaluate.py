import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from config import EMOTIONS, MODEL_PATH, OUTPUT_DIR
from inference import load_model
from utils import load_fer2013


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    args = parser.parse_args()
    _, _, (x_test, y_test) = load_fer2013()
    model = load_model(args.model_path)
    probabilities = model.predict(x_test, batch_size=128, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    print(classification_report(y_test, predictions, target_names=EMOTIONS, digits=4))
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1:          {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"Weighted F1:       {f1_score(y_test, predictions, average='weighted'):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Normalized Confusion Matrix")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
