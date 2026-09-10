from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "fer2013.csv"
MODEL_PATH = ROOT_DIR / "models" / "emotion_model.keras"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_SMOOTHING_ALPHA = 0.35
DEFAULT_SMOOTHING_WINDOW = 7
