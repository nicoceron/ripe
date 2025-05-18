# -*- coding: utf-8 -*-
"""
Fruit Quality Classification - Multi-class Classification Script
- Uses transfer learning with MobileNetV2
- Implements strict class balancing
- Adds color histogram features
- Multi-class classification (specific fruit+quality combinations)
- Excludes mixed quality fruits
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.api import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.api.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from collections import Counter
import random
from improved_training_monitoring import setup_enhanced_monitoring, add_activation_visualizer

# --- Use absolute paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Verify TensorFlow GPU (Metal) Initialization ---
print("TensorFlow Version:", tf.__version__)
physical_devices = tf.config.list_physical_devices()
print("Available Physical Devices:", physical_devices)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Metal GPU detected: {gpus}")
    # Configuration options can be set here if needed, e.g.:
    # try:
    #     tf.config.experimental.set_memory_growth(gpus[0], True)
    # except RuntimeError as e:
    #     print(e) # Memory growth must be set at program startup
else:
    print("No Metal GPU detected. TensorFlow will use the CPU.")
# -----------------------------------------------------

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- **Set Plot Style and Parameters** ---
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})
HIGH_DPI = 300 # DPI for saved figures

# --- Define Output Directory ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_files_multiclass")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All files will be saved in: {OUTPUT_DIR}/")

# --- Define directories ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data/Processed Images_Fruits")
bad_quality_dir = os.path.join(DATA_DIR, "Bad Quality_Fruits")
good_quality_dir = os.path.join(DATA_DIR, "Good Quality_Fruits")
# Mixed quality directory is excluded for this classification

# Check if data directories exist
if not os.path.isdir(bad_quality_dir) or not os.path.isdir(good_quality_dir):
    print("ERROR: Data directories not found. Please check paths.")
    print(f"Looking for: \n- {bad_quality_dir}\n- {good_quality_dir}")
    exit()

# --- Get class distribution to determine balanced sampling ---
def get_class_distribution():
    """Get number of samples per class"""
    class_counts = {}
    
    # Process each directory (only good and bad now)
    for directory, quality_suffix in [(bad_quality_dir, "Bad"), (good_quality_dir, "Good")]:
        for subfolder_name in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Set class name with proper naming
                if subfolder_name.endswith(f"_{quality_suffix}"):
                    class_name = subfolder_name  # Already has proper format like "Apple_Bad"
                else:
                    # Add suffix if not present
                    class_name = f"{subfolder_name}_{quality_suffix}"
                
                # Count valid images - using a more comprehensive approach
                image_files = []
                for f in os.listdir(subfolder_path):
                    # Check all possible image extensions, case-insensitive
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                        image_files.append(f)
                
                # Print debug info
                print(f"Found {len(image_files)} images in {subfolder_path} ({class_name})")
                
                # Count by specific class
                class_counts[class_name] = len(image_files)
    
    return class_counts

# --- Process directory with balanced sampling ---
def process_directory_balanced(directory, quality_suffix, file_paths, labels, samples_per_class):
    """Process directory with samples_per_class samples per class, keeping ALL real images and using augmentation only when needed"""
    if not os.path.isdir(directory):
        print(f"Warning: Directory does not exist - {directory}")
        return
    
    # Process each subfolder (each fruit type)
    for subfolder_name in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue
            
        # Set class name with proper naming
        if subfolder_name.endswith(f"_{quality_suffix}"):
            class_name = subfolder_name  # Already has proper format like "Apple_Bad"
        else:
            # Add suffix if not present
            class_name = f"{subfolder_name}_{quality_suffix}"
        
        # Get all valid image paths for this class
        image_paths = []
        for f in os.listdir(subfolder_path):
            # Check all possible image extensions, case-insensitive
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_paths.append(os.path.join(subfolder_path, f))
        
        available = len(image_paths)
        print(f"Processing {available} images for class {class_name}")
        
        # MODIFIED: Add ALL original images, no matter how many there are
        for path in image_paths:
            file_paths.append(path)
            labels.append(class_name)
        
        # Only augment if we need more samples to reach the target
        if available < samples_per_class:
            # Calculate how many more we need
            remaining = samples_per_class - available
            augmentation_dir = os.path.join(OUTPUT_DIR, "augmented", class_name)
            os.makedirs(augmentation_dir, exist_ok=True)
            
            print(f"  Class {class_name}: Using all {available} original images + generating {remaining} augmented images")
            
            # Calculate augmentations needed per original image
            augmentations_per_image = int(np.ceil(remaining / available))
            
            # Create augmented images
            added_augmentations = 0
            for i, path in enumerate(image_paths):
                if added_augmentations >= remaining:
                    break
                    
                # Load image to augment it
                img = cv2.imread(path)
                if img is None:
                    continue
                
                # Convert color format if needed
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                if img.shape[2] != 3:
                    continue
                
                # Generate augmented versions
                aug_images = augment_image(img, augmentations_per_image)
                
                # Add each augmented image
                for j, aug_img in enumerate(aug_images):
                    if added_augmentations >= remaining:
                        break
                        
                    # Save the augmented image
                    base_name = os.path.basename(path)
                    aug_filename = f"aug_{i}_{j}{os.path.splitext(base_name)[1]}"
                    aug_path = os.path.join(augmentation_dir, aug_filename)
                    cv2.imwrite(aug_path, aug_img)
                    
                    # Add to dataset
                    file_paths.append(aug_path)
                    labels.append(class_name)
                    added_augmentations += 1
            
            print(f"  Final count for {class_name}: {available + added_augmentations} images")
        else:
            print(f"  Class {class_name}: Using all {available} original images (no augmentation needed)")

# --- Set sample target dynamically based on largest class ---
class_counts = get_class_distribution()
print("\n--- Class Distribution in Original Dataset ---")
for cls, count in sorted(class_counts.items()):
    print(f"  - {cls}: {count}")

# Increase target samples per class - use max class size or a reasonable cap
max_samples = max(class_counts.values())
# Cap at 1500 samples per class to avoid excessive augmentation
SAMPLES_PER_CLASS = min(max_samples, 1500)  
print(f"\nTarget: {SAMPLES_PER_CLASS} samples per class (using ALL original images + augmentation when needed)")
print(f"Maximum available in any class: {max_samples}")

# --- Function to augment images ---
def augment_image(image, augmentation_factor):
    """Generate multiple augmented versions of an image"""
    augmented_images = []
    
    # Create a more aggressive augmentation pipeline for generating new images
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.3),
        layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    ])
    
    # Generate augmented versions
    for _ in range(augmentation_factor):
        # Apply augmentation to create a new variant
        augmented = image.copy()
        augmented = augmentation(tf.convert_to_tensor(augmented[np.newaxis, ...], dtype=tf.float32))[0].numpy().astype(np.uint8)
        augmented_images.append(augmented)
    
    return augmented_images

# --- Process image data with strict balancing and augmentation ---
file_paths = []
labels = []

process_directory_balanced(bad_quality_dir, "Bad", file_paths, labels, SAMPLES_PER_CLASS)
process_directory_balanced(good_quality_dir, "Good", file_paths, labels, SAMPLES_PER_CLASS)

if not file_paths:
    print("ERROR: No images found. Please check data paths.")
    exit()

print(f"\nTotal balanced dataset: {len(file_paths)} images")
print(f"Class distribution: {Counter(labels)}")

# --- Encode labels ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
class_names = label_encoder.classes_
num_classes = len(class_names)
print(f"Unique classes: {num_classes}")
print(f"Classes: {class_names}")

# --- Create DataFrame ---
combined_data = pd.DataFrame({'file_path': file_paths, 'label': encoded_labels, 'text_label': labels})

# --- Save class distribution plot ---
plt.figure(figsize=(12, 8))
class_counts = pd.Series(labels).value_counts().sort_index()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Distribution of Images by Class (Balanced)")
plt.xticks(rotation=75, ha='right')
plt.tight_layout()
distribution_plot_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(distribution_plot_path, dpi=HIGH_DPI)
plt.close()
print(f"Class distribution plot saved to: {distribution_plot_path}")

# --- Define Target Image Size ---
TARGET_SIZE = (224, 224) # Reduced size for memory efficiency
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)

# --- Show Image Examples ---
print("\nGenerating image examples plot...")
examples_plot_path = os.path.join(OUTPUT_DIR, "dataset_examples.png")
output_examples = []
try:
    num_examples_to_show = min(num_classes, 10)
    example_classes = np.random.choice(class_names, num_examples_to_show, replace=False)
    
    # Determine grid size based on number of classes
    grid_size = int(np.ceil(np.sqrt(num_examples_to_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(example_classes):
        # Get a random sample from this class
        sample_rows = combined_data[combined_data['text_label'] == class_name]
        if not sample_rows.empty:
            sample_row = sample_rows.iloc[np.random.randint(len(sample_rows))]
            img_path = sample_row['file_path']
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(class_name.replace("_", " "), fontsize=10)
                axes[i].axis('off')
                output_examples.append({'path': img_path, 'class': class_name})
    
    # Turn off any unused subplots
    for i in range(len(example_classes), len(axes)):
        axes[i].axis('off')
        
    fig.suptitle("Examples of Images from Dataset", fontsize=16)
    fig.tight_layout(pad=1.5, rect=[0, 0.03, 1, 0.95])
    plt.savefig(examples_plot_path, dpi=HIGH_DPI)
    print(f"Dataset examples plot saved to {examples_plot_path}")
    plt.close(fig)
except Exception as e:
    print(f"Error generating image examples plot: {e}")


# --- Function to extract color histograms ---
def extract_color_features(image_tensor):
    """Extract color, texture, and basic shape features from the image tensor."""
    # Convert the input tensor to a NumPy array for OpenCV
    image_np = image_tensor.numpy()
    
    # Ensure the numpy array is uint8, as expected by some cv2 functions
    if image_np.dtype != np.uint8:
        # This might happen if the input tensor wasn't cast correctly before py_function
        # Attempt to safely convert
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = (image_np * 255).astype(np.uint8)
        else:
             image_np = image_np.astype(np.uint8)

    # Check if the image has 3 channels (required for BGR2HSV)
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        print(f"Warning: extract_color_features received image with shape {image_np.shape}. Expected 3 channels. Skipping feature extraction.")
        # Return zeros or handle appropriately
        return np.zeros(117, dtype=np.float32) 
        
    # Convert to HSV color space (better for color analysis)
    # Use the numpy array from now on
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel with more bins for detail
    h_hist = cv2.calcHist([hsv_image], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [32], [0, 256])
    
    # Normalize the histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Add texture features using gradient magnitudes
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) # Use image_np
    gray_resized = cv2.resize(gray, (64, 64))  # Resize for faster processing
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    texture_hist = cv2.calcHist([np.uint8(mag)], [0], None, [16], [0, 256])
    texture_hist = cv2.normalize(texture_hist, texture_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Laplacian variance (blur measure)
    laplacian_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var() / 1000.0 # Scaled
    
    # Basic Shape Features using Contours
    shape_features = np.zeros(4) # aspect_ratio, extent, solidity, circularity
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 100: # Avoid tiny contours
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
        # print(f"Shape feature extraction failed: {e}")
        pass # Keep zeros if fails
        
    # Combine all features
    all_features = np.concatenate([
        h_hist, s_hist, v_hist, 
        texture_hist, 
        np.array([laplacian_var]), 
        shape_features 
    ]) # 32*3 + 16 + 1 + 4 = 96 + 16 + 1 + 4 = 117 features
    
    # Ensure float32 type for TensorFlow
    return all_features.astype(np.float32)

# --- Function to load and preprocess images and features for tf.data ---
def load_and_preprocess(file_path, label):
    # Read image file using tf.io
    image_bytes = tf.io.read_file(file_path)
    # Decode image, ensure 3 channels (RGB)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False) 
    
    # Handle potential grayscale or alpha channels explicitly if decode_image doesn't suffice
    # (decode_image usually handles common formats well)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    # We expect 3 channels now. If not, it's an issue.
    
    # Resize image
    image_resized = tf.image.resize(image, TARGET_SIZE)
    image_resized = tf.cast(image_resized, tf.uint8) # Cast for cv2 compatibility if needed below

    # --- Extract color features using numpy_function ---
    # cv2 functions require NumPy arrays, so we use tf.numpy_function
    # It's slightly less efficient than pure TF ops but necessary for cv2
    
    # Need to get numpy array from tensor to pass to cv2-based function
    # Use tf.py_function to wrap the feature extraction
    color_features = tf.py_function(
        func=extract_color_features, 
        inp=[image_resized], # Pass the uint8 resized image
        Tout=tf.float32
    )
    
    # Set shape for the features tensor - crucial for model building
    color_features.set_shape([117]) # Ensure the shape is known

    # Prepare image for EfficientNet (scaling is done in the model)
    # Cast image to float32 for the model
    image_final = tf.cast(image_resized, tf.float32)
    
    # Return a dictionary matching the model's input names
    return {'image_input': image_final, 'features_input': color_features}, label


# --- Prepare tf.data.Dataset ---
print("\nCreating tf.data pipeline...")

# Convert labels to one-hot encoding *before* creating dataset
y_one_hot = to_categorical(encoded_labels, num_classes=num_classes)

# Create the initial dataset from file paths and one-hot encoded labels
dataset = tf.data.Dataset.from_tensor_slices((file_paths, y_one_hot))

# Shuffle the dataset early (important for splitting)
dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42, reshuffle_each_iteration=False)

# --- Split the dataset ---
# No need for stratification here as the original list was balanced and shuffled
DATASET_SIZE = len(file_paths)
TRAIN_SIZE = int(0.7 * DATASET_SIZE)
VAL_SIZE = int(0.15 * DATASET_SIZE)
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE - VAL_SIZE # Ensure it sums up

train_dataset = dataset.take(TRAIN_SIZE)
val_test_dataset = dataset.skip(TRAIN_SIZE)
val_dataset = val_test_dataset.take(VAL_SIZE)
test_dataset = val_test_dataset.skip(VAL_SIZE)

print(f"Total dataset size: {DATASET_SIZE}")
print(f"Training set size: {TRAIN_SIZE}")
print(f"Validation set size: {VAL_SIZE}")
print(f"Test set size: {TEST_SIZE}")

# --- Apply preprocessing and batching ---
AUTOTUNE = tf.data.AUTOTUNE # Use tf.data runtime autotuning

# Use a function to configure datasets for performance
def configure_dataset(ds, shuffle=False, batch_size=32):
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        # Shuffle after mapping, buffer size can be smaller than full dataset
        ds = ds.shuffle(buffer_size=1000) 
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE) # Prefetch for performance
    return ds

# Set batch sizes
BATCH_SIZE_TRAIN = 32 # Keep original batch size for now
BATCH_SIZE_VAL_TEST = 32 

train_ds = configure_dataset(train_dataset, shuffle=True, batch_size=BATCH_SIZE_TRAIN)
val_ds = configure_dataset(val_dataset, batch_size=BATCH_SIZE_VAL_TEST)
test_ds = configure_dataset(test_dataset, batch_size=BATCH_SIZE_VAL_TEST)

print("tf.data pipeline created.")

# --- Verify dataset output shapes (optional debug step) ---
# print("\nSample batch from training dataset:")
# for batch_data, batch_labels in train_ds.take(1):
#     print("Image batch shape:", batch_data['image_input'].shape)
#     print("Features batch shape:", batch_data['features_input'].shape)
#     print("Labels batch shape:", batch_labels.shape)

# --- Define Enhanced Data Augmentation ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name="data_augmentation")

# --- Build model for multi-class classification ---
def build_model(input_shape, num_classes, handcrafted_features_dim):
    # Use EfficientNetB3 for higher accuracy potential
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Image input branch
    image_input = keras.Input(shape=input_shape, name='image_input')
    x = data_augmentation(image_input)
    x = tf.keras.applications.efficientnet.preprocess_input(x) # Use EfficientNet preprocessing
    
    # Apply base model and global pooling
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) # Add BN after GAP
    x = layers.Dropout(0.3)(x)  # Moderate dropout
    
    # Handcrafted features input branch
    features_input = keras.Input(shape=(handcrafted_features_dim,), name='features_input')
    feat = layers.Dense(128, activation='relu')(features_input)
    feat = layers.BatchNormalization()(feat)
    feat = layers.Dropout(0.3)(feat)
    
    # Combine both branches
    combined = layers.Concatenate()([x, feat])
    
    # Dense layers with swish activation
    x = layers.Dense(512, activation='swish')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer for multi-class classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model with two inputs
    model = keras.Model(inputs=[image_input, features_input], outputs=outputs)
    
    return model, base_model

# --- Create model ---
# Input shape is now defined earlier by TARGET_SIZE
color_features_dim = 117  # Updated dimension: 3*32 (HSV) + 16 (Texture) + 1 (LapVar) + 4 (Shape)

model, base_model = build_model(INPUT_SHAPE, num_classes, color_features_dim)
print("\nModel Summary:")
model.summary()

# --- Setup callbacks ---
callbacks, log_dir = setup_enhanced_monitoring(OUTPUT_DIR)

# Add early stopping with patience
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max'
)
callbacks.append(early_stopping)

# Add more aggressive model checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras"),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
callbacks.append(model_checkpoint)

# Add learning rate schedule with warm-up and proper decay
def lr_schedule(epoch, lr):
    # Warm-up phase (first 5 epochs) - gradually increase learning rate
    if epoch < 5:
        return initial_lr * ((epoch + 1) / 5)
    
    # After warm-up, use cosine decay
    if epoch < 30:  # Initial training phase
        decay_epochs = 30
        decay_steps = decay_epochs - 5
        rel_epoch = epoch - 5
        
        # Cosine decay with minimum of 1e-5
        decay = 0.5 * (1 + np.cos(np.pi * rel_epoch / decay_steps))
        return max(initial_lr * decay, 1e-5)
    
    # Fine-tuning phase (after epoch 30)
    else:
        # Use smaller learning rate with exponential decay
        ft_epoch = epoch - 30
        return fine_tuning_lr * np.exp(-0.05 * ft_epoch)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
callbacks.append(lr_scheduler)

# --- Define focal loss (if not already defined earlier or imported) ---
# Ensure FocalLoss class is defined or imported before this point
@tf.keras.utils.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.5, label_smoothing=0.03, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing
        num_classes = tf.shape(y_true)[-1]
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, y_true.dtype))
        
        # Apply focal loss
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

# --- Compile model (Initial Phase) ---
initial_lr = 5e-4  # 0.0005 as determined by improved multiclass hyperparameter tuning
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_lr,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True
)
focal_loss = FocalLoss(gamma=2.0, alpha=0.5, label_smoothing=0.02)

model.compile(
    optimizer=optimizer,
    loss=focal_loss,
    metrics=['accuracy']
)

# Create class weights for handling imbalance
class_weights = {}

# --- Train model (initial phase) ---
print("\nStarting initial training phase...")
history = model.fit(
    train_ds, # Use the tf.data.Dataset
    epochs=30,
    validation_data=val_ds, # Use the validation dataset
    callbacks=callbacks,
    # class_weight not typically needed with balanced tf.data pipeline, but kept if needed
    # class_weight=class_weights 
)

# --- Fine-tuning phase ---
print("\nStarting fine-tuning phase...")
base_model.trainable = True

# Only unfreeze the higher layers, keeping early feature extractors frozen
# For MobileNetV2, keep first 100 layers frozen for better transfer learning
for layer in base_model.layers[:100]:
    layer.trainable = False

# Use a lower learning rate for fine-tuning to prevent destroying features
fine_tuning_lr = initial_lr / 10
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tuning_lr),
    loss=focal_loss,
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_ds, # Use the tf.data.Dataset
    batch_size=None, # Batch size is controlled by the dataset
    epochs=100,  # More epochs for fine-tuning, early stopping will prevent overfitting
    initial_epoch=30,
    validation_data=val_ds, # Use the validation dataset
    callbacks=callbacks,
    # class_weight=class_weights
)

# --- Get the best saved model ---
print("\nLoading best model for evaluation...")
best_model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("model_epoch_") and f.endswith(".keras")]
best_model_file = None
best_acc = 0.0

for model_file in best_model_files:
    try:
        acc = float(model_file.split("val_acc_")[1].split(".keras")[0])
        if acc > best_acc:
            best_acc = acc
            best_model_file = model_file
    except:
        continue

if best_model_file:
    best_model_path = os.path.join(OUTPUT_DIR, best_model_file)
    print(f"Loading best model: {best_model_path} (validation accuracy: {best_acc:.4f})")
    model = tf.keras.models.load_model(
        best_model_path, 
        custom_objects={"FocalLoss": FocalLoss}
    )
else:
    print("No saved model found, using current model")

# --- Final test evaluation ---
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1) # Use the test dataset
print(f"Test accuracy: {test_acc:.4f}")

# --- Confusion matrix ---
print("\nGenerating final evaluation metrics...")

# Need to collect all true labels and predictions from the test_ds
y_true_list = []
y_pred_list = []

print("Making predictions on the test set...")
for batch_data, batch_labels in test_ds:
    preds = model.predict_on_batch(batch_data)
    y_pred_list.append(np.argmax(preds, axis=1))
    y_true_list.append(np.argmax(batch_labels.numpy(), axis=1)) # Get labels from one-hot

# Concatenate results from all batches
y_pred_classes = np.concatenate(y_pred_list)
y_test_classes_indices = np.concatenate(y_true_list)


cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
cm = confusion_matrix(y_test_classes_indices, y_pred_classes)
plt.figure(figsize=(16, 14))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Labels", fontsize=13)
plt.ylabel("True Labels", fontsize=13)
plt.title("Confusion Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout(pad=1.0)
plt.savefig(cm_plot_path, dpi=HIGH_DPI)
plt.close()
print(f"Confusion matrix saved to {cm_plot_path}")

# --- Classification report ---
print("\nClassification Report:")
report = classification_report(y_test_classes_indices, y_pred_classes, target_names=class_names, digits=3)
print(report)
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, 'w') as f:
    f.write("Classification Report\n=======================\n\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)
print(f"Classification report saved to {report_path}")

# --- Per-class accuracy visualization ---
class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(14, 8))
bar_colors = ['#3498db' if acc >= 0.8 else '#e74c3c' for acc in class_accuracies]
bars = plt.bar(class_names, class_accuracies, color=bar_colors)
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)

# Add accuracy values on top of bars
for bar, acc in zip(bars, class_accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{acc:.2f}',
        ha='center',
        fontsize=9
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_accuracies.png'), dpi=300)
plt.close()

# --- Group accuracy by fruit type ---
fruit_types = [class_name.split('_')[0] for class_name in class_names]
unique_fruits = list(set(fruit_types))

grouped_accuracies = {}
for fruit in unique_fruits:
    # Find indices for classes with this fruit
    indices = [i for i, cn in enumerate(class_names) if cn.startswith(fruit+'_')]
    
    # Calculate average accuracy for this fruit
    if indices:
        fruit_acc = np.mean([class_accuracies[i] for i in indices])
        grouped_accuracies[fruit] = fruit_acc

# Plot fruit-type accuracies
plt.figure(figsize=(10, 6))
sorted_fruits = sorted(grouped_accuracies.items(), key=lambda x: x[1], reverse=True)
fruits, accs = zip(*sorted_fruits)

plt.bar(fruits, accs, color='skyblue')
plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Overall Accuracy: {test_acc:.4f}')
plt.xlabel('Fruit Type')
plt.ylabel('Average Accuracy')
plt.title('Accuracy by Fruit Type')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fruit_type_accuracy.png'), dpi=300)
plt.close()

# --- Save best model ---
final_model_path = os.path.join(OUTPUT_DIR, "fruit_quality_multiclass_model.keras")
model.save(final_model_path)
print(f"\nSaved final model to {final_model_path}")

# --- Save class names & labels file ---
class_names_path = os.path.join(OUTPUT_DIR, 'class_names.npy')
labels_txt_path = os.path.join(OUTPUT_DIR, 'labels.txt')
np.save(class_names_path, class_names)
print(f"Saved {len(class_names)} class names to {class_names_path}")
with open(labels_txt_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"Labels file saved as {labels_txt_path}")

# --- Print final summary ---
print(f"\n--- Script execution completed ---")
print(f"Output files in folder: '{OUTPUT_DIR}'")
print(f"  - Keras Model: {os.path.basename(final_model_path)}")
print(f"  - Classification Report: {os.path.basename(report_path)}")
print(f"  - Confusion Matrix: {os.path.basename(cm_plot_path)}")
print(f"  - Class Accuracies: class_accuracies.png")
print("---")
print(f"FINAL TEST ACCURACY: {test_acc:.4f}")