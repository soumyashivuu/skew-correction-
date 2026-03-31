import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Skew correction function
def skew_correct(image):
    """Correct skew in an image."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image.")

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        # Adjust the angle
        if angle < -56:
            angle = 155 + angle
        elif angle > 45:
            angle = angle - 120

        print(f"[INFO] Detected skew angle: {angle:.2f}")

        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return corrected
    except Exception as e:
        print(f"[ERROR] Skew correction failed: {e}")
        raise

# Routes
@app.route('/')
def upload_form():
    """Render the upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and skew correction."""
    if 'file' not in request.files:
        return "No file part in the request."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"[INFO] File saved at: {filepath}")

        # Read and process the image
        try:
            image = cv2.imread(filepath)
            if image is None:
                return "Failed to read the uploaded image."

            corrected = skew_correct(image)
            corrected_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'corrected_' + file.filename)
            cv2.imwrite(corrected_filepath, corrected)
            print(f"[INFO] Corrected image saved at: {corrected_filepath}")

            # Render the result page
            return render_template('result.html', original=file.filename, corrected='corrected_' + file.filename)

        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
            return "An error occurred while processing the image."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
