import argparse

import cv2

from config import DEFAULT_MIN_CONFIDENCE, DEFAULT_SMOOTHING_ALPHA, DEFAULT_SMOOTHING_WINDOW, EMOTIONS, MODEL_PATH
from face_detector import HaarFaceDetector
from inference import load_model, predict_face, prediction_from_probabilities
from smoothing import PredictionSmoother


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--smoothing_alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--smoothing_window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.model_path)
    detector = HaarFaceDetector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    smoothers = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces, gray = detector.detect(frame)
            active = set()
            for x, y, w, h in faces:
                key = (round((x + w / 2) / 40), round((y + h / 2) / 40))
                active.add(key)
                smoother = smoothers.setdefault(key, PredictionSmoother(len(EMOTIONS), args.smoothing_alpha, args.smoothing_window))
                probabilities = smoother.update(predict_face(model, gray[y:y+h, x:x+w]))
                label, confidence, _ = prediction_from_probabilities(probabilities, args.min_confidence)
                text = f"{label} {confidence:.0%}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for key in set(smoothers) - active:
                del smoothers[key]
            cv2.imshow("Facial Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
