import os

class Config:
    # Base directory of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Directory for extracted face images (created automatically)
    DATA_DIR = os.path.join(BASE_DIR, 'data')  # Will contain 'male' and 'female' subfolders

    # Directory for trained models and preprocessed data (created automatically)
    TRAINING_DIR = os.path.join(BASE_DIR, 'training')  # Will contain .model files and .npy files

    # Path to OpenCV Haar Cascade classifier for face detection
    CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')  # Download from https://github.com/opencv/opencv/tree/master/data/haarcascades and place here

    # Path to SQLite database for storing face image paths
    DB_PATH = os.path.join(BASE_DIR, 'faces.db')  # Created automatically during data extraction

    # Model parameters
    IMG_SIZE = 32  # Input image size for the CNN (32x32 pixels)
    BATCH_SIZE = 32  # Batch size for training
    EPOCHS = 20  # Number of training epochs
    VALIDATION_SPLIT = 0.2  # Fraction of training data for validation
    TEST_SIZE = 0.2  # Fraction of data for testing

    # Categories for classification
    CATEGORIES = ['male', 'female']  # Gender categories
    LABEL_DICT = {cat: i for i, cat in enumerate(CATEGORIES)}  # Maps categories to integers

    # Video extraction settings
    VIDEO_PATH = os.path.join(BASE_DIR, 'My_video.mp4')  # Place your input video file here (e.g., rename to My_video.mp4)
    TIME_INTERVALS = [0, 10, 15, 20, 30]  # Time intervals in seconds to capture frames from video

    # Logging configuration
    LOG_LEVEL = 'INFO'  # Logging level (DEBUG, INFO, WARNING, ERROR)
    LOG_FILE = os.path.join(BASE_DIR, 'app.log')  # Log file path (created automatically)
