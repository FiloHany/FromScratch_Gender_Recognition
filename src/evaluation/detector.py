import cv2
import numpy as np
from ..config import Config
from ..logger import setup_logger
from ..training_code.model import ModelFactory

class GenderDetector:
    """
    Responsible for real-time gender detection.
    Single Responsibility: Only detection.
    """

    def __init__(self, config: Config = None, logger=None):
        self.config = config or Config()
        self.logger = logger or setup_logger('GenderDetector')
        self.model = None
        self.face_clsfr = cv2.CascadeClassifier(self.config.CASCADE_PATH)
        self.labels_dict = {v: k for k, v in self.config.LABEL_DICT.items()}
        self.color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  # Red for male, green for female

    def load_model(self, model_path: str):
        """Load the trained model."""
        from keras.models import load_model
        self.model = load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")

    def detect_from_webcam(self):
        """Perform real-time detection from webcam."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Cannot open webcam.")
            return

        self.logger.info("Starting webcam detection. Press 'q' to quit.")
        while True:
            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_clsfr.detectMultiScale(gray, 1.3, 3)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+w, x:x+w]
                resized = cv2.resize(face_img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, self.config.IMG_SIZE, self.config.IMG_SIZE, 1))
                result = self.model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(img, (x, y), (x+w, y+h), self.color_dict[label], 2)
                cv2.rectangle(img, (x, y-40), (x+w, y), self.color_dict[label], -1)
                cv2.putText(img, self.labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Gender Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Webcam detection stopped.")

    def detect_from_image(self, image_path: str):
        """Detect gender from a single image."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Cannot load image: {image_path}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_clsfr.detectMultiScale(gray, 1.3, 3)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+w, x:x+w]
            resized = cv2.resize(face_img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, self.config.IMG_SIZE, self.config.IMG_SIZE, 1))
            result = self.model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(img, (x, y), (x+w, y+h), self.color_dict[label], 2)
            cv2.putText(img, self.labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Gender Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.logger.info(f"Detection complete for {image_path}")
