# Bird's Type Classification with ResNet101

This project implements a bird species classification system using a fine-tuned ResNet101 model pretrained on ImageNet. The system identifies bird species from input images. A Flask-based web interface allows users to upload bird images and receive species classification results.

## Features
- Classifies bird species using a ResNet101-based model.
- Fine-tuned ResNet101 for accurate multi-class classification.
- Web interface for uploading images and viewing predictions.
- Modular scripts for training and inference.

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- Flask
- NumPy
- OpenCV
- Pillow
- See `requirements.txt` for a complete list.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/bird_classification.git
   cd bird_classification
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the `models/` directory contains the pretrained model (`bird_classifier_resnet101.h5`). Note: This is a placeholder; train the model or download pretrained weights.

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open a browser and navigate to `http://localhost:5000`.
3. Use the web interface to upload a bird image and view the predicted species.
4. Alternatively, run the classification script directly:
   ```bash
   python classify_bird.py --image path/to/bird.jpg
   ```
5. To train the model, prepare a dataset and run:
   ```bash
   python train_bird_classifier.py --dataset path/to/dataset
   ```

## Project Structure
```
bird_classification/
├── app.py                    # Flask web application
├── train_bird_classifier.py  # Training script
├── classify_bird.py          # Inference script
├── models/                   # Pretrained model weights
├── static/                   # Static files (CSS, uploads)
├── templates/                # HTML templates
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── docs/                     # Detailed documentation
```

## Documentation
See `docs/documentation.md` for detailed information on the model architecture, dataset preparation, training, and deployment.

## Notes
- The pretrained model weight (`bird_classifier_resnet101.h5`) is a placeholder. Train the model using `train_bird_classifier.py` with a labeled dataset.
- Dataset should contain images of birds organized by species (e.g., `dataset/sparrow/`, `dataset/eagle/`).
- This is a simplified implementation for educational purposes.

## License
MIT License. See `LICENSE` file for details.