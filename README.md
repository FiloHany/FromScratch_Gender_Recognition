# Scratch Gender Detection

A deep learning-based gender detection system using Convolutional Neural Networks (CNN) built with TensorFlow/Keras. The project extracts faces from video frames, labels them as male or female, trains a CNN model, and performs real-time gender detection via webcam.

## Features
- Face extraction from videos using OpenCV.
- Dataset creation with labeled male/female faces.
- CNN model training for binary classification (Male/Female).
- Real-time gender detection using webcam.
- SQLite database for storing face image paths.

## Prerequisites
- Python 3.7+
- OpenCV Haar Cascade for face detection (`haarcascade_frontalface_default.xml` - download from OpenCV GitHub if needed)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/scratch-gender-detection.git
   cd scratch-gender-detection
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the Haar Cascade classifier:
   - Place `haarcascade_frontalface_default.xml` in the root directory.
   - Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades

## Usage
For **production mode** and reproducibility, use the modular CLI scripts. For **easy experimentation** and quick prototyping, use the Jupyter notebook.

### Production Mode (Recommended)
Use the CLI scripts for scalable, reproducible execution:
1. **Extract Data**:
   - Place your video file (e.g., `My_video.mp4`) in the root directory.
   - Run: `python -m src.main extract`
   - This extracts faces at specified time intervals and saves them to `data/male/` and `data/female/` folders, with paths stored in `faces.db`.

2. **Preprocess Data**:
   - Run: `python -m src.main preprocess`
   - This loads images, preprocesses them, and saves to `training/data.npy` and `training/target.npy`.

3. **Train the Model**:
   - Run: `python -m src.main train`
   - This trains the CNN model and saves checkpoints in `./training/` as `model-{epoch}.model`.

4. **Real-Time Detection**:
   - For webcam: `python -m src.main detect`
   - For image: `python -m src.main detect --image-path path/to/image.jpg`
   - Press `q` to quit webcam feed.

### Easy Experimentation Mode
- Run `jupyter notebook ScratchGenderDetection.ipynb` for quick prototyping and experimentation.

## Project Structure
- `src/`: Modular Python package with core functionality.
  - `config.py`: Configuration settings.
  - `logger.py`: Logging setup.
  - `data_extractor.py`: Face extraction from videos.
  - `preprocessor.py`: Data loading and preprocessing.
  - `model.py`: CNN model definition with Factory pattern.
  - `trainer.py`: Model training logic.
  - `detector.py`: Real-time detection.
  - `main.py`: CLI entry point.
- `ScratchGenderDetection.ipynb`: Original Jupyter notebook.
- `data/`: Folder for extracted male/female face images.
- `training/`: Saved model files and preprocessed data (`.npy` files).
- `faces.db`: SQLite database for face image paths.
- `haarcascade_frontalface_default.xml`: OpenCV face detector.

## Model Architecture
- CNN with multiple Conv2D layers (32/64 filters), MaxPooling2D, Dropout for regularization.
- Input: 32x32 grayscale face images.
- Output: Softmax for binary classification (Male/Female).
- Trained for 20 epochs with categorical cross-entropy loss and Adam optimizer.

## Results
- The model achieves high accuracy on the extracted dataset.
- Real-time detection draws bounding boxes and labels on detected faces.

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with TensorFlow/Keras and OpenCV.
- Dataset extraction inspired by computer vision tutorials.
