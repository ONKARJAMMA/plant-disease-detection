import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('C:/Projects/PDD/plant_disease_model.h5')

# Class names - update this as per your model
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Resize the image to match the input size expected by the model (256x256 in this case)
    img = cv2.resize(img, (256, 256))
    
    # Normalize the image (optional, depending on how the model was trained)
    img = img / 255.0
    
    # Convert to the correct shape (1, 256, 256, 3)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded image to a temporary location
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Preprocess the image and predict the disease
        img = load_and_preprocess_image(image_path)
        prediction = model.predict(img)
        
        # Get the class with the highest probability
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Return the prediction result
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence)
        })

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
