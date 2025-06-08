# Bird's Type Classification Documentation

## Overview
This project implements a bird species classification system using a fine-tuned ResNet101 model pretrained on ImageNet. The system identifies bird species from input images. A Flask web interface allows users to upload images and view predicted species with confidence scores.

## Model Architecture
- **Base Model**: ResNet101 (pretrained on ImageNet, excluding top layers)
- **Input**: 224x224x3 RGB images
- **Modifications**:
  - Remove top fully connected layers.
  - Add GlobalAveragePooling2D, Dense(512, ReLU), Dropout(0.5), Dense(num_classes, softmax).
- **Output**: Probability distribution over bird species
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam (learning rate 1e-4)

## Dataset Preparation
- **Format**: Directory structure with subfolders for each species:
  ```
  dataset/
  ├── sparrow/
  ├── eagle/
  ├── owl/
  └── ...
  ```
- **Preprocessing**:
  - Resize images to 224x224.
  - Normalize pixel values to [0, 1].
  - Augment data (rotation, flip, zoom) for robustness.
- **Recommended Datasets**: CUB-200-2011, NABirds, or custom bird image collections.

## Training
1. Prepare a labeled dataset as described above.
2. Run the training script:
   ```bash
   python train_bird_classifier.py --dataset path/to/dataset
   ```
3. The script:
   - Loads and preprocesses the dataset.
   - Fine-tunes the ResNet101 model.
   - Saves the trained model to `models/bird_classifier_resnet101.h5`.

## Inference
- **Script (`classify_bird.py`)**:
  - Loads the pretrained model.
  - Preprocesses the input image.
  - Outputs the predicted species and confidence.
- **Web Interface (`app.py`)**:
  - Upload an image via the Flask interface.
  - Displays the predicted species and confidence score.

## Deployment
1. Install dependencies (`requirements.txt`).
2. Place the pretrained model in `models/`.
3. Run `app.py` to start the Flask server.
4. Access at `http://localhost:5000`.

## Implementation Details
- **Training (`train_bird_classifier.py`)**:
  - Uses TensorFlow's `ImageDataGenerator` for data loading and augmentation.
  - Freezes ResNet101 convolutional layers initially, then fine-tunes.
- **Inference (`classify_bird.py`)**:
  - Preprocesses images to match training conditions.
  - Returns the top predicted species and confidence.
- **Web Interface (`app.py`)**:
  - Flask application with routes for uploading images and displaying results.

## Future Improvements
- Incorporate attention mechanisms for better feature focus.
- Use ensemble models for improved accuracy.
- Add support for hierarchical classification (e.g., genus and species).
- Integrate with bird databases for additional metadata.

## References
- He, K., et al. (2016). Deep Residual Learning for Image Recognition.
- Chollet, F. (2017). Deep Learning with Python.