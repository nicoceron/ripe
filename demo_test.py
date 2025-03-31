import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Define directories (same as in training)
bad_quality_dir = '../data/Processed Images_Fruits/Bad Quality_Fruits'
good_quality_dir = '../data/Processed Images_Fruits/Good Quality_Fruits'
mixed_quality_dir = '../data/Processed Images_Fruits/Mixed Quality_Fruits'

# Load the labels
def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to find random image files
def get_random_images(root_dirs, num_images=5):
    all_image_paths = []
    
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(subdir, file))
    
    if len(all_image_paths) < num_images:
        num_images = len(all_image_paths)
        print(f"Warning: Only found {num_images} images")
    
    return random.sample(all_image_paths, num_images)

# Preprocess image for model input
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (100, 100))  # Same size as training
    return img.astype(np.float32)  # Convert to float32

# Load and initialize the TFLite model
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Run inference on an image
def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# Display image with predictions
def display_results(image_path, predictions, labels, save_dir=None):
    # Get top 3 predictions (or fewer if there aren't 3 classes)
    num_to_display = min(3, len(predictions))
    top_indices = np.argsort(predictions)[-num_to_display:][::-1]
    top_probs = predictions[top_indices]
    top_labels = [labels[i] for i in top_indices]
    
    # Display image and predictions
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract expected label from path for verification
    folder_parts = image_path.split(os.sep)
    if 'Bad Quality_Fruits' in image_path:
        expected_label = folder_parts[-2]  # e.g., Apple_Bad
    elif 'Good Quality_Fruits' in image_path:
        expected_label = folder_parts[-2]  # e.g., Apple_Good
    elif 'Mixed Quality_Fruits' in image_path:
        expected_label = f"{folder_parts[-2]}_mixed"  # e.g., Apple_mixed
    else:
        expected_label = "Unknown"
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Image: {expected_label}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top_labels))
    bars = plt.barh(y_pos, top_probs)
    
    # Color the bar green if it matches the expected label
    for i, label in enumerate(top_labels):
        if label == expected_label:
            bars[i].set_color('green')
    
    plt.yticks(y_pos, top_labels)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    
    # Show full path at the bottom
    path_info = os.path.basename(image_path)
    plt.figtext(0.5, 0.01, f"File: {path_info}", 
                ha="center", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for figtext
    
    # Save the figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(save_dir, f"{basename}_prediction.png")
        plt.savefig(save_path)
        print(f"Saved result to: {save_path}")
    
    plt.show()

def main():
    try:
        # Load labels
        label_file = 'labels.txt'
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Labels file not found: {label_file}")
        
        labels = load_labels(label_file)
        print(f"Loaded {len(labels)} labels:")
        for i, label in enumerate(labels):
            print(f"  {i}: {label}")
        
        # Load TFLite model
        model_path = 'fruit_quality_model.tflite'
        interpreter = load_tflite_model(model_path)
        print("Model loaded successfully")
        
        # Ask user how many images to test
        try:
            num_images = int(input("How many random images would you like to test? (default: 5): ") or "5")
        except ValueError:
            num_images = 5
            print("Invalid input, using default: 5 images")
        
        # Get random images
        image_paths = get_random_images(
            [bad_quality_dir, good_quality_dir, mixed_quality_dir], 
            num_images=num_images
        )
        
        print(f"\nSelected {len(image_paths)} random images for testing")
        
        # Create a directory to save results
        save_dir = 'test_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # Process each image and display results
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Preprocess the image
            processed_image = preprocess_image(image_path)
            
            # Run inference
            predictions = run_inference(interpreter, processed_image)
            
            # Get the highest probability prediction
            top_index = np.argmax(predictions)
            confidence = predictions[top_index]
            predicted_label = labels[top_index]
            
            print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
            
            # Display detailed results with visualization
            display_results(image_path, predictions, labels, save_dir)
            
        print(f"\nAll results have been saved to the '{save_dir}' directory")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()