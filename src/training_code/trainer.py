import os
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from ..config import Config
from ..logger import setup_logger
from .model import ModelFactory

class ModelTrainer:
    """
    Responsible for training the model.
    Single Responsibility: Only training.
    """

    def __init__(self, config: Config = None, logger=None):
        self.config = config or Config()
        self.logger = logger or setup_logger('ModelTrainer')

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed data and split into train/test."""
        data_path = os.path.join(self.config.TRAINING_DIR, 'data.npy')
        target_path = os.path.join(self.config.TRAINING_DIR, 'target.npy')

        if not os.path.exists(data_path) or not os.path.exists(target_path):
            raise FileNotFoundError("Preprocessed data not found. Run preprocessor first.")

        data = np.load(data_path)
        target = np.load(target_path)

        # Normalize and reshape
        data = data / 255.0
        data = data.reshape(data.shape[0], self.config.IMG_SIZE, self.config.IMG_SIZE, 1)
        target = to_categorical(target, num_classes=len(self.config.CATEGORIES))

        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=self.config.TEST_SIZE, random_state=42
        )

        self.logger.info(f"Data loaded: {len(train_data)} train, {len(test_data)} test samples.")
        return train_data, test_data, train_target, test_target

    def train(self):
        """Train the model."""
        self.logger.info("Starting model training.")
        train_data, test_data, train_target, test_target = self.load_data()

        model = ModelFactory.create_model('cnn', self.config).get_model()

        checkpoint = ModelCheckpoint(
            os.path.join(self.config.TRAINING_DIR, 'model-{epoch:03d}.model'),
            monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
        )

        history = model.fit(
            train_data, train_target,
            epochs=self.config.EPOCHS,
            callbacks=[checkpoint],
            validation_split=self.config.VALIDATION_SPLIT
        )

        # Evaluate on test set
        loss, accuracy = model.evaluate(test_data, test_target)
        self.logger.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

        self.logger.info("Model training complete.")
        return history
