#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo test script for Fruit Quality Classification using TFLite model
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import argparse
import sys
from pathlib import Path

# Get the absolute path of the demo directory
DEMO_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Configuration
MODEL_PATH = str(DEMO_DIR / "models/fruit_quality_model.tflite")
LABELS_PATH = str(DEMO_DIR / "models/labels.txt")
SAMPLES_DIR = str(DEMO_DIR / "samples")
TARGET_SIZE = (224, 224)

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
    """Load and preprocess an image for TFLite prediction"""
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
    img_tensor = img_resized_rgb.astype(np.float32)
    
    return img_tensor, color_features, original_img

def load_labels(label_path):
    """Load class labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def predict_with_tflite(model_path, image_path, label_path):
    """Predict image using TFLite model"""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model input details for debugging
    print("\nModel Input Details:")
    for i, detail in enumerate(input_details):
        print(f"Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")
    
    # Load and preprocess the image
    img_tensor, color_features, original_img = load_and_preprocess_image(image_path)
    
    # Set model inputs - handle input tensor shapes correctly
    if len(input_details) >= 2:
        # Important: The model actually expects features first, then image 
        # (based on the shape details printed above)
        for i, detail in enumerate(input_details):
            # Check actual shape of the input tensor
            shape = detail['shape']
            if shape[1] == 117:  # This is the features input
                print(f"Setting input {i} with features shape: {color_features.shape}")
                interpreter.set_tensor(detail['index'], np.expand_dims(color_features, axis=0))
            elif len(shape) == 4:  # This is the image input
                print(f"Setting input {i} with image shape: {img_tensor.shape}")
                interpreter.set_tensor(detail['index'], np.expand_dims(img_tensor, axis=0))
    else:
        # Handle single input model (just image)
        detail = input_details[0]
        print(f"Single input model, setting input with shape: {detail['shape']}")
        interpreter.set_tensor(detail['index'], np.expand_dims(img_tensor, axis=0))
    
    # Run inference
    interpreter.invoke()
    
    # Get the prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Load labels
    class_names = load_labels(label_path)
    
    # Get top 3 predictions
    top_indices = np.argsort(output_data[0])[::-1][:3]
    top_probs = output_data[0][top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(DEMO_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
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
        results_dir, 
        f"prediction_{os.path.basename(image_path)}"
    )
    plt.savefig(result_filename)
    plt.show()
    
    print(f"\nTop 3 Predictions for {os.path.basename(image_path)}:")
    for cls, prob in zip(top_classes, top_probs):
        print(f"  {cls.replace('_', ' ')}: {prob:.4f}")
    
    print(f"\nPrediction visualization saved to {result_filename}")
    
    return top_classes[0], top_probs[0]

def list_sample_images():
    """List all sample images available in the samples directory"""
    samples = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        samples.extend(list(Path(SAMPLES_DIR).glob(ext)))
    
    if not samples:
        print("No sample images found in the samples directory.")
        return None
    
    print("\nAvailable sample images:")
    for i, sample_path in enumerate(samples):
        print(f"  {i+1}. {sample_path.name}")
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Test fruit quality classification model (TFLite)')
    parser.add_argument('--image_path', type=str, help='Path to the image file')
    parser.add_argument('--sample', type=int, help='Index of sample image to use (see list with --list)')
    parser.add_argument('--list', action='store_true', help='List available sample images')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the TFLite model file')
    parser.add_argument('--labels', type=str, default=LABELS_PATH, help='Path to the labels file')
    args = parser.parse_args()
    
    if args.list:
        list_sample_images()
        return
    
    if args.sample is not None:
        samples = list_sample_images()
        if not samples:
            return
        
        if args.sample < 1 or args.sample > len(samples):
            print(f"Error: Sample index must be between 1 and {len(samples)}")
            return
        
        image_path = str(samples[args.sample - 1])
    elif args.image_path:
        image_path = args.image_path
    else:
        samples = list_sample_images()
        if not samples:
            return
        print("\nNo image specified, using first sample image.")
        image_path = str(samples[0])
    
    try:
        predict_with_tflite(args.model, image_path, args.labels)
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main() 