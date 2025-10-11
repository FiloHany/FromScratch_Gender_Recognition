import argparse
import os
from .config import Config
from .logger import setup_logger
from .data_code.data_extractor import DataExtractor
from .data_code.preprocessor import DataPreprocessor
from .training_code.trainer import ModelTrainer
from .evaluation.detector import GenderDetector

def main():
    parser = argparse.ArgumentParser(description='Gender Detection System')
    parser.add_argument('command', choices=['extract', 'preprocess', 'train', 'detect'],
                        help='Command to run')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model for detection')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to image for detection (alternative to webcam)')

    args = parser.parse_args()

    config = Config()
    logger = setup_logger()

    if args.command == 'extract':
        extractor = DataExtractor(config, logger)
        extractor.run_extraction()

    elif args.command == 'preprocess':
        preprocessor = DataPreprocessor(config, logger)
        preprocessor.preprocess_and_save()

    elif args.command == 'train':
        trainer = ModelTrainer(config, logger)
        trainer.train()

    elif args.command == 'detect':
        detector = GenderDetector(config, logger)
        if args.model_path:
            detector.load_model(args.model_path)
        else:
            # Load latest model
            model_files = [f for f in os.listdir(config.TRAINING_DIR) if f.endswith('.model')]
            if model_files:
                latest_model = max(model_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
                detector.load_model(os.path.join(config.TRAINING_DIR, latest_model))
            else:
                logger.error("No trained model found.")
                return

        if args.image_path:
            detector.detect_from_image(args.image_path)
        else:
            detector.detect_from_webcam()

if __name__ == '__main__':
    main()
