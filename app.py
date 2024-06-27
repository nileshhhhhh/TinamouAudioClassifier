from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import librosa
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('DLmodel.h5')

# Function to extract MFCC features from an audio file
def get_mfcc(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    mfcc_scaled_features = mfcc_scaled_features.reshape(1, -1)
    return mfcc_scaled_features

# Function to get prediction
def preds(predictions):
    pred_class_labels = np.argmax(predictions, axis=1)
    if pred_class_labels == [0]:
        return "Brown Tinamou"
    elif pred_class_labels == [1]:
        return "Cinereous Tinamou"
    else:
        return "Great Tinamou"

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        mfcc_features = get_mfcc(file_path)
        predictions = model.predict(mfcc_features)
        prediction_label = preds(predictions)
        os.remove(file_path)  # Optionally remove the uploaded file after prediction
        return render_template('result.html', prediction=prediction_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
