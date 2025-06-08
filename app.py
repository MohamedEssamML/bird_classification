from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (placeholder)
classifier = tf.keras.models.load_model('models/bird_classifier_resnet101.h5')
classes = ['sparrow', 'eagle', 'owl']  # Update based on dataset

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess and classify
        image = Image.open(filepath).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = classifier.predict(image)[0]
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return render_template('index.html', 
                             result=f'Predicted Species: {predicted_class} (Confidence: {confidence:.2%})',
                             image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)