import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

def classify_image(model, image_path, classes):
    # Load and preprocess image
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image)[0]
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    args = parser.parse_args()
    
    classifier = tf.keras.models.load_model('models/bird_classifier_resnet101.h5')
    classes = ['sparrow', 'eagle', 'owl']  # Update based on dataset
    predicted_class, confidence = classify_image(classifier, args.image, classes)
    print(f'Predicted Species: {predicted_class} (Confidence: {confidence:.2%})')