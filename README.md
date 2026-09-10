# Facial Emotion Detection

A practical FER-2013 facial emotion recognition project using TensorFlow/Keras and OpenCV. The project is designed as a strong, reproducible baseline for real-time webcam inference rather than a claim of perfect emotion understanding.

## What changed in v2

- **More regularized CNN:** global average pooling reduces the large fully-connected parameter bottleneck.
- **Label smoothing:** makes training less overconfident on noisy FER-2013 labels.
- **Class-balanced training:** automatically computes class weights from the training split, helping underrepresented emotions such as Disgust.
- **Stronger augmentation:** adds small rotations, shifts, zoom, shear, reflection padding and horizontal flips.
- **Better preprocessing:** histogram equalization is applied before live inference to reduce sensitivity to lighting.
- **Temporal smoothing:** live predictions use a short probability-history smoother, reducing frame-to-frame flicker.
- **Uncertain state:** low-confidence predictions are shown as `Uncertain` instead of presenting a weak softmax maximum as a fact.
- **Stronger evaluation:** reports accuracy, macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1 and a normalized confusion matrix.
- **Modular inference:** detector, preprocessing, model inference and smoothing are separated so each component can be replaced independently.

## Project structure

```text
facial-emotion-detection/
├── data/
│   ├── .gitkeep
│   └── fer2013.csv                 # ignored by Git
├── models/
│   ├── .gitkeep
│   └── emotion_model.keras         # ignored by Git
├── outputs/
│   └── .gitkeep
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── face_detector.py
│   ├── inference.py
│   ├── live_detection.py
│   ├── model.py
│   ├── smoothing.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Download FER-2013 as `data/fer2013.csv`.

## Train

```bash
python src/train.py
```

For a quick smoke test:

```bash
python src/train.py --sample_size 2000 --epochs 2
```

The best checkpoint is written to `models/emotion_model.keras`.

## Evaluate

```bash
python src/evaluate.py
```

The evaluation uses the FER-2013 `PrivateTest` split and prints class-level metrics plus balanced accuracy and macro F1. A normalized confusion matrix is saved to `outputs/confusion_matrix_normalized.png`.

## Live detection

```bash
python src/live_detection.py
```

Useful options:

```bash
python src/live_detection.py --min_confidence 0.50 --smoothing_window 7
```

Press **Q** to quit.

### Detection limitations

The current detector is still OpenCV's Haar cascade because it is lightweight and requires no extra model download. It works best with reasonably front-facing, visible faces. A future accuracy-focused version should benchmark a modern detector (for example, an OpenCV DNN detector) and face alignment before comparing emotion-model performance.

## Dataset and model notes

FER-2013 contains noisy, crowd-sourced facial-expression labels. Emotion recognition is probabilistic and should not be interpreted as reading a person's true internal emotional state.

The trained model file is intentionally ignored by Git. For distribution, use Git LFS or a GitHub Release/model registry rather than storing large binary checkpoints in normal Git history.

## License

MIT License. See `LICENSE`.
