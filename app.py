import os
import uuid
import time
import numpy as np
import nibabel as nib
import joblib
import cv2
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use Agg backend to avoid Tkinter issues
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from utils import resize_and_skull_strip, extract_features  # Ensure these are implemented in your utils.py
import base64
from threading import Thread

app = Flask(__name__)

# Load pre-trained Random Forest model
model = joblib.load('trained_model_RF.pkl')

# Directory to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Schedule file deletion after processing to avoid clutter
def schedule_file_deletion(path, delay=60):
    def delete_file():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    Thread(target=delete_file, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']
    image = request.files['image']

    if disease not in ['MS', 'CSVD']:
        return jsonify({'error': 'Invalid disease type'})

    # Create a unique filename to prevent collisions
    unique_filename = f"{uuid.uuid4().hex}_{image.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    image.save(image_path)

    # Load and preprocess the NIfTI image
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
    except Exception as e:
        return jsonify({'error': f"Error loading image: {str(e)}"})

    # Convert to individual slices
    slices = [data[:, :, i] for i in range(data.shape[2])]

    # Preprocess slices (e.g., skull strip + resize)
    processed = resize_and_skull_strip(slices)

    # Extract features from the processed slices
    features = extract_features(processed)

    # Predict the probabilities and class
    probs = model.predict_proba(features)
    predictions = np.argmax(probs, axis=1)

    # Final prediction: majority vote
    final_prediction = np.bincount(predictions).argmax()

    # Confidence: average probability of the majority class
    confidence = np.mean(probs[:, final_prediction]) * 100
    confidence = np.clip(confidence, 0, 100)

    # Set the label and box color based on the prediction
    if final_prediction == 1:
        label = "MS (Multiple Sclerosis)"
        box_color = (0, 255, 0)  # Green
    elif final_prediction == 0:
        label = "CSVD (Small Vessel Disease)"
        box_color = (255, 0, 0)  # Red
    else:
        label = "Healthy Brain"
        box_color = (255, 255, 255)  # White

    # Visualize the middle slice
    mid_slice_idx = len(slices) // 2
    img_slice = slices[mid_slice_idx]

    # Normalize to 8-bit
    img_normalized = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 90, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Default to no bounding boxes if no contours are found
    draw_boxes = True
    if len(contours) == 0 or max(cv2.contourArea(cnt) for cnt in contours) < 200:
        label = "No Disease Found"
        box_color = (255, 255, 255)
        draw_boxes = False

    # Prepare the processed image for output
    img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
    if draw_boxes:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 100:  # Ignore very small boxes
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_color, 2)

    # Convert processed image to base64
    _, img_encoded = cv2.imencode('.png', img_bgr)
    processed_img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Save the input slice image for display
    input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_input.png")
    plt.imshow(img_slice, cmap='gray')
    plt.axis('off')
    plt.savefig(input_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Convert input image to base64 for the response
    with open(input_img_path, "rb") as input_image_file:
        input_img_base64 = base64.b64encode(input_image_file.read()).decode('utf-8')

    # Schedule deletion of the uploaded and input images
    schedule_file_deletion(image_path)
    schedule_file_deletion(input_img_path)

    # Return results in JSON format
    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 2),
        'input_image_url': f"data:image/png;base64,{input_img_base64}",
        'processed_image_url': f"data:image/png;base64,{processed_img_base64}"
    })


if __name__ == '__main__':
    app.run(debug=True)
