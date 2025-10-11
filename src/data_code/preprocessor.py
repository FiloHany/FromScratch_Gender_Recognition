import os
import cv2
import numpy as np
from typing import Tuple
from ..config import Config
from ..logger import setup_logger

class DataPreprocessor:
    """
    Responsible for loading, preprocessing, and saving dataset.
    Follows Single Responsibility: Only preprocessing.
    """

    def __init__(self, config: Config = None, logger=None):
        self.config = config or Config()
        self.logger = logger or setup_logger('DataPreprocessor')

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from data directories, preprocess, and return data and targets.
        """
        data = []
        target = []
        cascade = cv2.CascadeClassifier(self.config.CASCADE_PATH)

        for category in self.config.CATEGORIES:
            folder_path = os.path.join(self.config.DATA_DIR, category)
            if not os.path.exists(folder_path):
                self.logger.warning(f"Folder not found: {folder_path}")
                continue

            img_names = os.listdir(folder_path)
            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for f in faces:
                    x, y, w, h = f
                    sub_face = img[y:y + h, x:x + w]
                    gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (self.config.IMG_SIZE, self.config.IMG_SIZE))
                    data.append(resized)
                    target.append(self.config.LABEL_DICT[category])

        data = np.array(data)
        target = np.array(target)
        self.logger.info(f"Loaded {len(data)} samples.")
        return data, target

    def save_data(self, data: np.ndarray, target: np.ndarray):
        """Save preprocessed data to npy files."""
        if not os.path.exists(self.config.TRAINING_DIR):
            os.makedirs(self.config.TRAINING_DIR)

        np.save(os.path.join(self.config.TRAINING_DIR, 'data.npy'), data)
        np.save(os.path.join(self.config.TRAINING_DIR, 'target.npy'), target)
        self.logger.info("Data saved to npy files.")

    def preprocess_and_save(self):
        """Main method to preprocess and save data."""
        self.logger.info("Starting data preprocessing.")
        data, target = self.load_data()
        self.save_data(data, target)
        self.logger.info("Data preprocessing complete.")
