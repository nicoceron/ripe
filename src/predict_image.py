#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fruit Quality Classification - Single Image Prediction Script
This script loads the trained model and makes predictions on a single image.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras

# --- Configuration ---
# Use absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TARGET_SIZE = (224, 224)  # Must match the size used for training

# If environment variable for models directory is set, use it
if 'MODELS_DIR' in os.environ:
    MODEL_DIR = os.environ['MODELS_DIR']

print(f"Using models directory: {MODEL_DIR}")

# --- Define FocalLoss for model loading ---
@tf.keras.utils.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.5, label_smoothing=0.03, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_true)[-1]
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, y_true.dtype))
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1.0 - y_pred, self.gamma) * y_true
        fl = self.alpha * weight * ce
        reduced_fl = tf.reduce_sum(fl, axis=-1)
        return reduced_fl
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing
        })
        return config

# --- Helper Functions ---
def extract_color_features(image):
    """Extract color, texture, and basic shape features from the image"""
    # Ensure the image is BGR format (as expected by OpenCV)
    if len(image.shape) != 3 or image.shape[2] != 3:
        print(f"Warning: extract_color_features received image with shape {image.shape}. Expected 3 channels.")
        return np.zeros(117, dtype=np.float32)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    h_hist = cv2.calcHist([hsv_image], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [32], [0, 256])
    
    # Normalize the histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Texture features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (64, 64))
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    texture_hist = cv2.calcHist([np.uint8(mag)], [0], None, [16], [0, 256])
    texture_hist = cv2.normalize(texture_hist, texture_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Laplacian variance (blur measure)
    laplacian_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var() / 1000.0
    
    # Shape Features
    shape_features = np.zeros(4)  # aspect_ratio, extent, solidity, circularity
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 100:  # Avoid tiny contours
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                rect_area = w*h
                extent = float(area)/rect_area if rect_area > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4*np.pi*(area/(perimeter**2)) if perimeter > 0 else 0
                shape_features = np.array([aspect_ratio, extent, solidity, circularity])
    except Exception as e:
        pass
    
    # Combine all features
    all_features = np.concatenate([
        h_hist, s_hist, v_hist,
        texture_hist,
        np.array([laplacian_var]),
        shape_features
    ])
    
    return all_features.astype(np.float32)

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Keep original image for display
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img, TARGET_SIZE)
    
    # Extract color features
    color_features = extract_color_features(img_resized)
    
    # Preprocess for model
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_resized_rgb, dtype=tf.float32)
    
    return img_tensor, color_features, original_img

def predict_image(model, image_path, class_names):
    """Predict the class of a single image"""
    # Load and preprocess the image
    img_tensor, color_features, original_img = load_and_preprocess_image(image_path)
    
    # Prepare inputs for the model (match the expected format)
    img_batch = tf.expand_dims(img_tensor, axis=0)
    features_batch = tf.expand_dims(color_features, axis=0)
    
    # Make prediction
    prediction = model.predict({
        'image_input': img_batch,
        'features_input': features_batch
    }, verbose=0)
    
    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[::-1][:3]
    top_probs = prediction[0][top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Display results
    plt.figure(figsize=(10, 6))
    
    # Show the image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Show prediction results
    plt.subplot(1, 2, 2)
    plt.barh(range(3), top_probs)
    plt.yticks(range(3), [c.replace('_', ' ') for c in top_classes])
    plt.xlabel('Probability')
    plt.title('Top 3 Predictions')
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # Save and show results
    result_filename = os.path.join(
        os.path.dirname(image_path), 
        f"prediction_{os.path.basename(image_path)}"
    )
    plt.savefig(result_filename)
    plt.show()
    
    print(f"\nTop 3 Predictions for {os.path.basename(image_path)}:")
    for cls, prob in zip(top_classes, top_probs):
        print(f"  {cls.replace('_', ' ')}: {prob:.4f}")
    
    print(f"\nPrediction visualization saved to {result_filename}")
    
    return top_classes[0], top_probs[0]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict fruit quality from a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default=None, help='Path to the model file (optional)')
    args = parser.parse_args()
    
    # Determine model path
    model_path = args.model if args.model else os.path.join(MODEL_DIR, "fruit_quality_multiclass_model.keras")
    
    # Load class names
    class_names_path = os.path.join(os.path.dirname(model_path), "..", "output_files_multiclass", "class_names.npy")
    if not os.path.exists(class_names_path):
        class_names_path = os.path.join(os.path.dirname(model_path), "class_names.npy")
    
    try:
        class_names = np.load(class_names_path, allow_pickle=True)
        print(f"Loaded {len(class_names)} class names.")
    except:
        print(f"Could not load class names from {class_names_path}")
        # Fallback to common class names if file not found
        class_names = [
            'Apple_Bad', 'Apple_Good', 'Banana_Bad', 'Banana_Good',
            'Guava_Bad', 'Guava_Good', 'Lime_Bad', 'Lime_Good',
            'Orange_Bad', 'Orange_Good', 'Pomegranate_Bad', 'Pomegranate_Good'
        ]
        print(f"Using default class names: {class_names}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"FocalLoss": FocalLoss}
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    try:
        predict_image(model, args.image_path, class_names)
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main() 