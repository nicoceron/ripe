import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.api.models import load_model
from keras.api.layers import Dense, Flatten, Dropout, Input
from keras.api.applications import MobileNetV3Large
from keras.api.models import Model

# -------------------------------
# 1. Custom Preprocessing Function
# -------------------------------
def preprocess_fn(image):
    image = np.array(image)
    if image.dtype != np.uint8 or image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    # Convert RGB -> BGR (OpenCV uses BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert to LAB, apply CLAHE on the L-channel
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Otsu's thresholding for segmentation
    gray = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented = cv2.bitwise_and(image_clahe, image_clahe, mask=thresh)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    return segmented_rgb

def preprocess_input(image_path):
    # Load image (BGR -> RGB)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to expected input size
    image = cv2.resize(image, (256, 256))
    processed = preprocess_fn(image)
    processed = processed.astype('float32') / 255.0
    image_batch = np.expand_dims(processed, axis=0)
    return image_batch, image  # return both preprocessed batch and original image

# -------------------------------
# 2. (Optional) Rebuild Model Architecture
# -------------------------------
# Use this if loading the complete model fails.
def build_model(input_shape=(256,256,3), num_classes=15):
    inputs = Input(shape=input_shape)
    base_model = MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# -------------------------------
# 3. Load the Saved Model
# -------------------------------
# Try loading with compile=False and custom_objects to avoid the positional argument error.
try:
    model = load_model('fruitvision_best.h5', compile=False,
                       custom_objects={'MobileNetV3Large': MobileNetV3Large})
except Exception as e:
    print("Error loading complete model:", e)
    print("Rebuilding model and loading weights instead...")
    model = build_model(input_shape=(256,256,3), num_classes=15)
    model.load_weights('fruitvision_best.h5')

# -------------------------------
# 4. Select Sample Images from Data
# -------------------------------
my_classes = [
    'Apple_Fresh', 'Apple_Rotten', 'Apple_Formalin-mixed',
    'Banana_Fresh', 'Banana_Rotten', 'Banana_Formalin-mixed',
    'Grape_Fresh', 'Grape_Rotten', 'Grape_Formalin-mixed',
    'Mango_Fresh', 'Mango_Rotten', 'Mango_Formalin-mixed',
    'Orange_Fresh', 'Orange_Rotten', 'Orange_Formalin-mixed'
]
base_dir = './data'
sample_image_paths = []
sample_true_labels = []

for fruit_class in my_classes:
    class_dir = os.path.join(base_dir, fruit_class)
    # List image files; adjust extensions as needed
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        sample_image_paths.append(os.path.join(class_dir, files[0]))
        sample_true_labels.append(fruit_class)

# -------------------------------
# 5. Make Predictions and Display Results
# -------------------------------
plt.figure(figsize=(15, 10))
for i, img_path in enumerate(sample_image_paths):
    input_img, original_img = preprocess_input(img_path)
    prediction = model.predict(input_img)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = my_classes[predicted_index]
    
    plt.subplot(3, 5, i + 1)
    plt.imshow(original_img)
    plt.title(f"True: {sample_true_labels[i]}\nPred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
