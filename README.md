# Facial Emotion Detection

A modular facial emotion recognition system built with TensorFlow/Keras and OpenCV, trained around the FER-2013 benchmark and designed for real-time webcam inference.

> **Important:** This project predicts visual expression patterns from faces. It does not reliably determine a person's internal emotional state, intent, or mental health.

## Highlights

- Regularized CNN with global average pooling
- Label smoothing for noisy FER-2013 labels
- Class-balanced training for uneven emotion frequencies
- Stronger image augmentation
- Lighting normalization during live detection
- Temporal probability smoothing to reduce flicker
- Explicit `Uncertain` output for low-confidence predictions
- Evaluation with accuracy, macro F1, weighted F1 and balanced accuracy
- Normalized confusion matrix generation
- Modular detector → preprocessing → inference → smoothing pipeline

## Architecture

```text
Webcam
  ↓
Face detection
  ↓
Grayscale + lighting normalization
  ↓
48 × 48 preprocessing
  ↓
CNN emotion classifier
  ↓
Emotion probabilities
  ↓
Confidence threshold
  ↓
Temporal smoothing
  ↓
Stable emotion / Uncertain
```

## Project structure

```text
facial-emotion-detection/
├── data/
│   └── .gitkeep                  # FER-2013 CSV is not committed
├── models/
│   └── .gitkeep                  # trained weights are not committed
├── outputs/
│   └── .gitkeep                  # generated evaluation artifacts
├── src/
│   ├── config.py                 # paths and shared configuration
│   ├── evaluate.py               # test-set metrics and confusion matrix
│   ├── face_detector.py          # face detection
│   ├── inference.py              # model loading and prediction
│   ├── live_detection.py         # webcam application
│   ├── model.py                  # CNN architecture
│   ├── smoothing.py              # temporal prediction smoothing
│   ├── train.py                  # training pipeline
│   └── utils.py                  # dataset and preprocessing helpers
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the FER-2013 CSV at:

```text
data/fer2013.csv
```

The dataset is intentionally excluded from Git.

## Train

Full training:

```bash
python src/train.py
```

Quick smoke test:

```bash
python src/train.py --sample_size 2000 --epochs 2
```

The best checkpoint is written to `models/emotion_model.keras`.

### Training improvements

The training pipeline uses class-balanced weights, label smoothing, early stopping, learning-rate reduction, and stronger geometric augmentation. These choices are intended to improve robustness rather than simply maximize training accuracy.

## Evaluate

```bash
python src/evaluate.py
```

Evaluation uses FER-2013's `PrivateTest` split and reports:

- accuracy
- macro F1
- weighted F1
- balanced accuracy
- per-emotion precision, recall and F1
- normalized confusion matrix

The confusion matrix is saved to:

```text
outputs/confusion_matrix_normalized.png
```

## Live webcam detection

```bash
python src/live_detection.py
```

Optional controls:

```bash
python src/live_detection.py --min_confidence 0.50 --smoothing_window 7
```

Press **Q** to quit.

The live pipeline uses a lightweight Haar cascade detector, histogram equalization, confidence filtering and temporal probability smoothing. Haar detection remains the main practical limitation for difficult poses, occlusion and poor lighting.

## Model limitations

FER-2013 contains noisy, crowd-sourced expression labels and seven broad expression categories. Real-world performance can vary substantially with lighting, camera quality, face angle, occlusion, demographics and dataset shift.

A future accuracy-focused version should benchmark a modern face detector, face alignment, stronger CNN/transfer-learning backbones and calibration on a held-out validation set.

## Reproducibility

The repository does **not** commit the FER-2013 dataset or trained model weights. This keeps the Git history lightweight and avoids redistributing dataset contents without checking their terms.

For a portfolio/demo release, a trained checkpoint can be distributed separately through GitHub Releases, Git LFS, or another model registry after checking the relevant dataset/model terms.

## License

MIT License. See `LICENSE`.

The MIT license applies to the code in this repository. FER-2013 is a separate dataset with its own terms.
