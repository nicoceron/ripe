import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.api.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from collections import Counter

# --- Use absolute paths based on script location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "output_files_multiclass") # Directory where the model and class names are saved
DATA_DIR = os.path.join(PROJECT_ROOT, "data/Processed Images_Fruits") # Base directory containing Good/Bad Quality folders
BAD_QUALITY_DIR = os.path.join(DATA_DIR, 'Bad Quality_Fruits')
GOOD_QUALITY_DIR = os.path.join(DATA_DIR, 'Good Quality_Fruits')
TARGET_SIZE = (224, 224) # Must match the size used for training
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 32 # Batch size for evaluation
EVAL_OUTPUT_DIR = os.path.join(MODEL_DIR, "full_dataset_evaluation") # Subdir for full eval results
HIGH_DPI = 300

# If environment variables are set, use them
if 'MODELS_DIR' in os.environ:
    MODEL_DIR = os.environ['MODELS_DIR']

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
print(f"Using model directory: {MODEL_DIR}")
print(f"Full dataset evaluation results will be saved in: {EVAL_OUTPUT_DIR}")

# --- Helper Functions (Copied/adapted from training script) ---

# Need FocalLoss definition for loading the model
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

def extract_color_features(image_tensor):
    """Extract color, texture, and basic shape features from the image tensor."""
    image_np = image_tensor.numpy()
    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = (image_np * 255).astype(np.uint8)
        else:
             image_np = image_np.astype(np.uint8)

    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        print(f"Warning: extract_color_features received image with shape {image_np.shape}. Expected 3 channels. Skipping.")
        return np.zeros(117, dtype=np.float32) 
        
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [32], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (64, 64))
    sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    texture_hist = cv2.calcHist([np.uint8(mag)], [0], None, [16], [0, 256])
    texture_hist = cv2.normalize(texture_hist, texture_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    laplacian_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var() / 1000.0
    
    shape_features = np.zeros(4)
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 100:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h if h > 0 else 0
                rect_area = w*h
                extent = float(area)/rect_area if rect_area > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(cnt,True)
                circularity = 4*np.pi*(area/(perimeter**2)) if perimeter > 0 else 0
                shape_features = np.array([aspect_ratio, extent, solidity, circularity])
    except Exception as e:
        pass 
        
    all_features = np.concatenate([
        h_hist, s_hist, v_hist, texture_hist, np.array([laplacian_var]), shape_features 
    ])
    return all_features.astype(np.float32)

def load_and_preprocess(file_path, label):
    image_bytes = tf.io.read_file(file_path)
    # Explicitly decode to 3 channels (RGB)
    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False, dtype=tf.uint8)
    except tf.errors.InvalidArgumentError as e:
        # Handle potential decoding errors (e.g., corrupted file)
        print(f"Warning: Could not decode image {file_path}. Error: {e}. Skipping.")
        # Return dummy data that matches expected structure but can be filtered later if needed
        # Or design the model/downstream logic to handle potential zero/empty inputs.
        # For simplicity here, we'll return zeros.
        dummy_image = tf.zeros([TARGET_SIZE[0], TARGET_SIZE[1], 3], dtype=tf.float32)
        dummy_features = tf.zeros([117], dtype=tf.float32)
        # Note: The label is still passed through.
        return {'image_input': dummy_image, 'features_input': dummy_features}, label

    # Set shape explicitly after decoding, as decode_image might not infer it.
    image.set_shape([None, None, 3])

    # Resize the RGB image
    image_resized_rgb = tf.image.resize(image, TARGET_SIZE)
    image_resized_rgb_uint8 = tf.cast(image_resized_rgb, tf.uint8) # Cast for BGR conversion and py_function

    # Convert RGB to BGR for OpenCV function (extract_color_features expects BGR)
    image_resized_bgr_uint8 = tf.reverse(image_resized_rgb_uint8, axis=[-1])

    # Extract color features using the BGR image
    color_features = tf.py_function(
        func=extract_color_features, 
        inp=[image_resized_bgr_uint8], # Pass BGR uint8
        Tout=tf.float32
    )
    color_features.set_shape([117]) 

    # Prepare the RGB image for the model (EfficientNet expects RGB)
    image_final_rgb_float32 = tf.cast(image_resized_rgb, tf.float32) 
    
    # Apply EfficientNet preprocessing *within* the load function (expects RGB)
    image_preprocessed = tf.keras.applications.efficientnet.preprocess_input(image_final_rgb_float32)

    return {'image_input': image_preprocessed, 'features_input': color_features}, label

def configure_dataset(ds, batch_size=32):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE) 
    return ds

def get_all_filepaths_labels(base_dir_good, base_dir_bad):
    """Gets all image file paths and labels from the good and bad directories."""
    all_file_paths = []
    all_labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    print("Scanning directories for all image files...")
    
    for directory, quality_suffix in [(base_dir_bad, "Bad"), (base_dir_good, "Good")]:
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found - {directory}")
            continue
            
        for subfolder_name in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Construct class name (e.g., Apple_Bad)
                fruit_name = subfolder_name.replace(f"_{quality_suffix}", "")
                class_name = f"{fruit_name}_{quality_suffix}"
                
                image_count = 0
                for fname in os.listdir(subfolder_path):
                    if fname.lower().endswith(valid_extensions):
                        all_file_paths.append(os.path.join(subfolder_path, fname))
                        all_labels.append(class_name)
                        image_count += 1
                print(f"  Found {image_count} images in {subfolder_path} (Class: {class_name})")

    print(f"Total images found across all classes: {len(all_file_paths)}")
    if not all_file_paths:
        print("ERROR: No image files found in the specified directories.")
        exit()
        
    print("Full dataset class distribution:", Counter(all_labels))
    return all_file_paths, all_labels

# --- Main Evaluation Logic ---

# 1. Load Model
model_path = os.path.join(MODEL_DIR, "fruit_quality_multiclass_model.keras")
print(f"\nLoading trained model from: {model_path}")
if not os.path.exists(model_path):
    print("ERROR: Model file not found!")
    exit()
    
try:
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"FocalLoss": FocalLoss}
    )
    print("Model loaded successfully.")
    # model.summary() # Optional: print model summary
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Load Class Names
class_names_path = os.path.join(MODEL_DIR, 'class_names.npy')
print(f"Loading class names from: {class_names_path}")
if not os.path.exists(class_names_path):
    print("ERROR: Class names file not found!")
    exit()
class_names = np.load(class_names_path, allow_pickle=True)
num_classes = len(class_names)
print(f"Loaded {num_classes} class names: {class_names}")

# 3. Get All Filepaths and Labels
all_filepaths, all_labels_text = get_all_filepaths_labels(GOOD_QUALITY_DIR, BAD_QUALITY_DIR)

# 4. Encode Labels
label_encoder = LabelEncoder()
label_encoder.fit(class_names) # Fit on the loaded class names to ensure consistency
all_labels_encoded = label_encoder.transform(all_labels_text)
all_labels_one_hot = to_categorical(all_labels_encoded, num_classes=num_classes)

# 5. Create tf.data.Dataset
full_dataset = tf.data.Dataset.from_tensor_slices((all_filepaths, all_labels_one_hot))

# 6. Configure Dataset
full_eval_ds = configure_dataset(full_dataset, batch_size=BATCH_SIZE)
print("\nFull dataset prepared for evaluation.")

# 7. Evaluate Model
print("\nEvaluating model on the full dataset...")
eval_loss, eval_accuracy = model.evaluate(full_eval_ds, verbose=1)
print(f"\n--- Full Dataset Evaluation ---:")
print(f"  Loss: {eval_loss:.4f}")
print(f"  Accuracy: {eval_accuracy:.4f}")

# 8. Generate Predictions and Detailed Report (Optional but Recommended)
print("\nGenerating predictions for detailed report...")
y_pred_list = []
y_true_list = []

# Iterate through the dataset to get predictions and true labels
for batch_data, batch_labels in full_eval_ds:
    preds = model.predict_on_batch(batch_data)
    y_pred_list.append(np.argmax(preds, axis=1))
    y_true_list.append(np.argmax(batch_labels.numpy(), axis=1))

# Concatenate results
y_pred_classes = np.concatenate(y_pred_list)
y_true_classes_indices = np.concatenate(y_true_list)

# Ensure shapes match
print(f"Shape of true labels: {y_true_classes_indices.shape}")
print(f"Shape of predicted labels: {y_pred_classes.shape}")

# Classification Report
print("\nClassification Report (Full Dataset):")
report = classification_report(y_true_classes_indices, y_pred_classes, target_names=class_names, digits=3)
print(report)
report_path = os.path.join(EVAL_OUTPUT_DIR, "classification_report_full_dataset.txt")
with open(report_path, 'w') as f:
    f.write("Classification Report (Full Dataset)\n===================================\n\n")
    f.write(f"Overall Accuracy: {eval_accuracy:.4f}\n")
    f.write(f"Overall Loss: {eval_loss:.4f}\n\n")
    f.write(report)
print(f"Full dataset classification report saved to {report_path}")

# Confusion Matrix
print("\nGenerating confusion matrix for full dataset...")
cm = confusion_matrix(y_true_classes_indices, y_pred_classes)
cm_plot_path = os.path.join(EVAL_OUTPUT_DIR, "confusion_matrix_full_dataset.png")

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Labels", fontsize=13)
plt.ylabel("True Labels", fontsize=13)
plt.title("Confusion Matrix (Full Dataset)", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout(pad=1.0)
plt.savefig(cm_plot_path, dpi=HIGH_DPI)
plt.close()
print(f"Full dataset confusion matrix saved to {cm_plot_path}")

print("\n--- Full dataset evaluation script finished ---") 