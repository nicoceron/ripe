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
import time
from tensorflow.keras import layers # type: ignore
from ai_edge_litert.interpreter import Interpreter

# --- Use absolute paths based on script location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "output_files_multiclass") # Directory where the Keras model and class names are saved
DATA_DIR = os.path.join(PROJECT_ROOT, "data/Processed Images_Fruits") # Base directory containing Good/Bad Quality folders
BAD_QUALITY_DIR = os.path.join(DATA_DIR, 'Bad Quality_Fruits')
GOOD_QUALITY_DIR = os.path.join(DATA_DIR, 'Good Quality_Fruits')
TARGET_SIZE = (224, 224) # Must match the size used for training
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 32 # Batch size for evaluation (can adjust for TFLite if needed)
TFLITE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tflite_export") # Directory to save TFLite model and results
HIGH_DPI = 300

# If environment variables are set, use them
if 'MODELS_DIR' in os.environ:
    MODEL_DIR = os.environ['MODELS_DIR']

os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)
print(f"Using model directory: {MODEL_DIR}")
print(f"TFLite model and evaluation results will be saved in: {TFLITE_OUTPUT_DIR}")

# --- Helper Functions (Copied/adapted from training/evaluation script) ---

# Need FocalLoss definition for loading the Keras model
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
        
    # Input is expected to be BGR uint8 here
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

# --- Build Model Architecture (without Augmentation) for Weight Transfer ---
def build_inference_model(input_shape, num_classes, handcrafted_features_dim):
    # Replicates the trained model structure *without* the data_augmentation layer
    
    # Base model setup (must match training)
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights=None # Load weights later
    )
    base_model.trainable = False # Match state during saving (though weights are fixed anyway)
    
    # Image input branch (NO data_augmentation)
    image_input = keras.Input(shape=input_shape, name='image_input')
    x = tf.keras.applications.efficientnet.preprocess_input(image_input) # Preprocess directly
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dropout(0.3)(x)
    
    # Handcrafted features input branch
    features_input = keras.Input(shape=(handcrafted_features_dim,), name='features_input')
    feat = layers.Dense(128, activation='relu')(features_input)
    feat = layers.BatchNormalization()(feat)
    feat = layers.Dropout(0.3)(feat)
    
    # Combine both branches
    combined = layers.Concatenate()([x, feat])
    
    # Dense layers (must match training architecture)
    x = layers.Dense(512, activation='swish')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the inference model
    inference_model = keras.Model(inputs=[image_input, features_input], outputs=outputs)
    return inference_model

def load_and_preprocess_for_tflite(file_path, label):
    """Loads and preprocesses image and features for TFLite evaluation (CORRECTED)."""
    image_bytes = tf.io.read_file(file_path)
    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False, dtype=tf.uint8)
    except tf.errors.InvalidArgumentError as e:
        print(f"Warning: Could not decode image {file_path}. Error: {e}. Returning zeros.")
        # Return dummy data matching the expected structure
        dummy_image = tf.zeros([TARGET_SIZE[0], TARGET_SIZE[1], 3], dtype=tf.float32) # Match expected model input type
        dummy_features = tf.zeros([117], dtype=tf.float32)
        return (dummy_image, dummy_features), label

    image.set_shape([None, None, 3])
    image_resized_rgb = tf.image.resize(image, TARGET_SIZE) # Output is float32

    # --- Prepare image for FEATURE extraction (needs BGR uint8) ---
    image_resized_rgb_uint8 = tf.cast(image_resized_rgb, tf.uint8)
    image_resized_bgr_uint8 = tf.reverse(image_resized_rgb_uint8, axis=[-1])
    color_features = tf.py_function(
        func=extract_color_features,
        inp=[image_resized_bgr_uint8],
        Tout=tf.float32
    )
    color_features.set_shape([117])

    # --- Prepare image for MODEL input ---
    # The TFLite model has preprocess_input baked in.
    # It expects Float32 input, typically in [0, 255] range BEFORE its internal step.
    # tf.image.resize already outputs float32 in [0, 255] range if input is uint8.
    image_for_model_input = image_resized_rgb # Use the float32 output from resize directly

    # ***** REMOVE THE EXTERNAL PREPROCESSING STEP *****
    # image_preprocessed = tf.keras.applications.efficientnet.preprocess_input(image_for_model_input)

    # Return the image data the TFLite model's internal preprocessing step expects
    return (image_for_model_input, color_features), label

def configure_tflite_dataset(ds, batch_size=32):
    """Configure tf.data pipeline for TFLite evaluation."""
    AUTOTUNE = tf.data.AUTOTUNE
    # Use the TFLite-specific preprocessing function
    ds = ds.map(load_and_preprocess_for_tflite, num_parallel_calls=AUTOTUNE)
    # Batch the dataset AFTER mapping
    ds = ds.batch(batch_size) 
    ds = ds.prefetch(buffer_size=AUTOTUNE) 
    return ds

def get_all_filepaths_labels(base_dir_good, base_dir_bad):
    """Gets all image file paths and labels from the good and bad directories."""
    # (Identical to the function in evaluate_on_full_dataset.py)
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
                fruit_name = subfolder_name.replace(f"_{quality_suffix}", "")
                class_name = f"{fruit_name}_{quality_suffix}"
                image_count = 0
                for fname in os.listdir(subfolder_path):
                    if fname.lower().endswith(valid_extensions):
                        all_file_paths.append(os.path.join(subfolder_path, fname))
                        all_labels.append(class_name)
                        image_count += 1
                # print(f"  Found {image_count} images in {subfolder_path} (Class: {class_name})") # Verbose
    print(f"Total images found across all classes: {len(all_file_paths)}")
    if not all_file_paths:
        print("ERROR: No image files found.")
        exit()
    # print("Full dataset class distribution:", Counter(all_labels)) # Verbose
    return all_file_paths, all_labels

# --- Main Conversion and Evaluation Logic ---

# 1. Load Keras Model
keras_model_path = os.path.join(MODEL_DIR, "fruit_quality_multiclass_model.keras")
print(f"\nLoading Keras model from: {keras_model_path}")
if not os.path.exists(keras_model_path):
    print("ERROR: Keras Model file not found!")
    exit()
try:
    model = tf.keras.models.load_model(
        keras_model_path, 
        custom_objects={"FocalLoss": FocalLoss}
    )
    print("Keras model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

# 2. Load Class Names (Needed BEFORE building inference model)
class_names_path = os.path.join(MODEL_DIR, 'class_names.npy')
print(f"Loading class names from: {class_names_path}")
if not os.path.exists(class_names_path):
    print("ERROR: Class names file not found! Cannot proceed.")
    exit()
class_names = np.load(class_names_path, allow_pickle=True)
num_classes = len(class_names)
print(f"Loaded {num_classes} class names.")

# 3. Create Inference Model Structure and Load Weights
print("\nBuilding inference model structure...")
# Use num_classes derived from loaded file
color_features_dim = 117 # From config
inference_model = build_inference_model(INPUT_SHAPE, num_classes, color_features_dim)

print("Setting weights from loaded model to inference model...")
inference_model.set_weights(model.get_weights())
print("Weights transferred successfully.")

# Clean up the loaded model with augmentation to save memory
del model 

# 4. Convert the Inference Model (without augmentation) to TFLite
print("\nConverting inference model to TFLite with dynamic batch size...")

# Note: We still use TensorFlow's converter for model conversion
# LiteRT is only used for model inference, not conversion
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define the expected input signature for dynamic batch size
input_signature = [
    tf.TensorSpec(shape=[None] + list(inference_model.inputs[0].shape[1:]), dtype=inference_model.inputs[0].dtype, name=inference_model.inputs[0].name.split(':')[0]),
    tf.TensorSpec(shape=[None] + list(inference_model.inputs[1].shape[1:]), dtype=inference_model.inputs[1].dtype, name=inference_model.inputs[1].name.split(':')[0])
]

converter.input_signature = input_signature

# Optional: Specify supported types if needed for specific hardware (e.g., float16)
# converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
    print(f"TFLite conversion successful. Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"Error during TFLite conversion: {e}")
    exit()

# 5. Save TFLite Model
tflite_model_path = os.path.join(TFLITE_OUTPUT_DIR, "fruit_quality_multiclass_model.tflite")
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to: {tflite_model_path}")

# Save class names to labels.txt for use in inference scripts
labels_path_txt = os.path.join(TFLITE_OUTPUT_DIR, "labels.txt")
with open(labels_path_txt, 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')
print(f"Labels saved to: {labels_path_txt}")
print(f"Using {num_classes} class names for evaluation.")

# 6. Prepare Full Dataset for TFLite Evaluation
all_filepaths, all_labels_text = get_all_filepaths_labels(GOOD_QUALITY_DIR, BAD_QUALITY_DIR)
label_encoder = LabelEncoder()
label_encoder.fit(class_names)
all_labels_encoded = label_encoder.transform(all_labels_text)
# We need integer labels for comparison later, not one-hot for TFLite eval typically
# Keep the integer labels: all_labels_encoded 

# Create dataset with filepaths and *integer* labels
full_dataset = tf.data.Dataset.from_tensor_slices((all_filepaths, all_labels_encoded))

# Configure the dataset - IMPORTANT: Use batch size of 1 for TFLite evaluation
tflite_eval_ds = configure_tflite_dataset(full_dataset, batch_size=1)
print("\nFull dataset prepared for TFLite evaluation.")

# 7. Evaluate TFLite Model
print("\nEvaluating TFLite model...")

# Load the TFLite model and allocate tensors - using LiteRT instead of tf.lite
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug: Print input/output details
print("\nTFLite Input Details:", input_details)
print("TFLite Output Details:", output_details)

# Find the correct input indices based on names
image_input_index = -1
features_input_index = -1
image_input_dtype = None
features_input_dtype = None

for detail in input_details:
    # Adjust name matching to account for TFLite converter renaming
    if 'image_input' in detail['name']:
        image_input_index = detail['index']
        image_input_dtype = detail['dtype']
        print(f"  Found image input: {detail['name']} at index {image_input_index}")
    elif 'features_input' in detail['name']:
        features_input_index = detail['index']
        features_input_dtype = detail['dtype']
        print(f"  Found features input: {detail['name']} at index {features_input_index}")

# Verify indices were found
if image_input_index == -1 or features_input_index == -1:
    print("ERROR: Could not find image or features input by name.")
    print("Actual Input Details:", input_details) # Print details again if error
    exit()

# Assuming the first output is the correct one
try:
    output_index = output_details[0]['index']
    print(f"\nIdentified Input Indices: Image={image_input_index}, Features={features_input_index}")
    print(f"Identified Output Index: {output_index}")
    print(f"Expected Input tensor types: Image={image_input_dtype}, Features={features_input_dtype}")
except (IndexError, KeyError) as e:
    print(f"Error getting TFLite output details: {e}")
    print("Output Details:", output_details)
    exit()

y_pred_list = []
y_true_list = []

start_time = time.time()
processed_samples = 0
total_samples = len(all_filepaths)

# Iterate through the dataset one sample at a time
for batch_data_tuple, batch_labels in tflite_eval_ds:
    # batch_data_tuple contains (image_batch, features_batch) with batch size 1
    image_batch, features_batch = batch_data_tuple
    
    # Ensure the batch data types match the interpreter's expected input types
    image_sample_np = image_batch.numpy().astype(image_input_dtype)
    features_sample_np = features_batch.numpy().astype(features_input_dtype)

    # Set the value of the input tensors
    interpreter.set_tensor(image_input_index, image_sample_np)
    interpreter.set_tensor(features_input_index, features_sample_np)

    # Run inference
    interpreter.invoke()

    # Get the results
    preds = interpreter.get_tensor(output_index)
    y_pred_list.append(np.argmax(preds[0]))  # Get prediction for single sample
    y_true_list.append(batch_labels.numpy()[0])  # Get true label for single sample
    
    processed_samples += 1
    if processed_samples % 500 == 0:
        print(f"  Processed {processed_samples}/{total_samples} images ({processed_samples/total_samples*100:.1f}%)...")

end_time = time.time()
eval_duration = end_time - start_time
print(f"TFLite evaluation finished in {eval_duration:.2f} seconds.")

# Concatenate results - no need for concatenate since we're collecting individual predictions
y_pred_classes = np.array(y_pred_list)
y_true_classes_indices = np.array(y_true_list)

# Calculate overall accuracy
tflite_accuracy = accuracy_score(y_true_classes_indices, y_pred_classes)
print(f"\n--- TFLite Model Evaluation (Full Dataset) ---:")
print(f"  Accuracy: {tflite_accuracy:.4f}")

# 8. Generate Report and Confusion Matrix for TFLite

# Classification Report
print("\nClassification Report (TFLite - Full Dataset):")
report = classification_report(y_true_classes_indices, y_pred_classes, target_names=class_names, digits=3)
print(report)
tflite_report_path = os.path.join(TFLITE_OUTPUT_DIR, "tflite_classification_report_full_dataset.txt")
with open(tflite_report_path, 'w') as f:
    f.write("TFLite Classification Report (Full Dataset)\n=========================================\n\n")
    f.write(f"Overall Accuracy: {tflite_accuracy:.4f}\n\n")
    f.write(report)
print(f"TFLite classification report saved to {tflite_report_path}")

# Confusion Matrix
print("\nGenerating TFLite confusion matrix for full dataset...")
cm = confusion_matrix(y_true_classes_indices, y_pred_classes)
tflite_cm_plot_path = os.path.join(TFLITE_OUTPUT_DIR, "tflite_confusion_matrix_full_dataset.png")

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Labels", fontsize=13)
plt.ylabel("True Labels", fontsize=13)
plt.title("Confusion Matrix (TFLite - Full Dataset)", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout(pad=1.0)
plt.savefig(tflite_cm_plot_path, dpi=HIGH_DPI)
plt.close()
print(f"TFLite confusion matrix saved to {tflite_cm_plot_path}")

print("\n--- TFLite conversion and evaluation script finished ---") 