import cv2
import os
import sqlite3
from typing import List
from ..config import Config
from ..logger import setup_logger

class DataExtractor:
    """
    Responsible for extracting faces from video frames and saving them to folders and database.
    Follows Single Responsibility Principle: Only handles data extraction.
    """

    def __init__(self, config: Config = None, logger=None):
        self.config = config or Config()
        self.logger = logger or setup_logger('DataExtractor')
        self._setup_database()
        self._setup_directories()

    def _setup_database(self):
        """Initialize SQLite database for storing face image paths."""
        self.conn = sqlite3.connect(self.config.DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                category TEXT
            )
        ''')
        self.conn.commit()
        self.logger.info("Database setup complete.")

    def _setup_directories(self):
        """Create necessary directories for data storage."""
        for category in self.config.CATEGORIES:
            folder_path = os.path.join(self.config.DATA_DIR, category)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                self.logger.info(f"Created directory: {folder_path}")

    def capture_frames(self, video_path: str, time_intervals: List[int]) -> List:
        """
        Capture frames from video at specified time intervals.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return frames

        for interval in time_intervals:
            frame_number = int(interval * cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                self.logger.info(f"Captured frame at {interval}s")
            else:
                self.logger.warning(f"Failed to capture frame at {interval}s")
        cap.release()
        return frames

    def extract_and_save_faces(self, frames: List, cascade_path: str):
        """
        Detect faces in frames, save them to folders and database.
        """
        face_detector = cv2.CascadeClassifier(cascade_path)
        if face_detector.empty():
            self.logger.error(f"Failed to load cascade: {cascade_path}")
            return

        count = 0
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                # For simplicity, assign to 'male' or 'female' alternately or based on some logic
                # In production, this would be manual labeling
                category = self.config.CATEGORIES[count % len(self.config.CATEGORIES)]
                filename = f'face_{count}.jpg'
                filepath = os.path.join(self.config.DATA_DIR, category, filename)
                cv2.imwrite(filepath, face_image)

                # Save to DB
                self.cursor.execute('INSERT INTO faces (image_path, category) VALUES (?, ?)', (filepath, category))
                self.conn.commit()

                self.logger.info(f"Saved face: {filepath}")
                count += 1

        self.logger.info(f"Extracted and saved {count} faces.")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")

    def run_extraction(self):
        """Main method to run the extraction process."""
        self.logger.info("Starting data extraction.")
        frames = self.capture_frames(self.config.VIDEO_PATH, self.config.TIME_INTERVALS)
        self.extract_and_save_faces(frames, self.config.CASCADE_PATH)
        self.close()
        self.logger.info("Data extraction complete.")
