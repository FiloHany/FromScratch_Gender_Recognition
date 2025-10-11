from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from ..config import Config
from ..logger import setup_logger

class ModelFactory:
    """
    Factory pattern for creating different model types.
    Open/Closed: Easy to add new model types without modifying existing code.
    """

    @staticmethod
    def create_model(model_type: str = 'cnn', config: Config = None):
        if model_type == 'cnn':
            return CNNModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class BaseModel(ABC):
    """
    Abstract base class for models, following Interface Segregation.
    """

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def compile(self):
        pass

class CNNModel(BaseModel):
    """
    CNN model for gender detection.
    Single Responsibility: Only model definition.
    """

    def __init__(self, config: Config = None, logger=None):
        self.config = config or Config()
        self.logger = logger or setup_logger('CNNModel')
        self.model = None

    def build(self):
        """Build the CNN architecture."""
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 1), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.config.CATEGORIES), activation='softmax'))

        self.logger.info("CNN model built.")

    def compile(self):
        """Compile the model."""
        if self.model is None:
            self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        self.logger.info("Model compiled.")

    def get_model(self):
        """Return the compiled model."""
        if self.model is None:
            self.compile()
        return self.model
