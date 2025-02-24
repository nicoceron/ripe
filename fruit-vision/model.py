import os
import cv2
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense, Flatten, Dropout, Input
from keras.api.models import Model
from keras.api.applications import MobileNetV3Large
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------
# 1. Custom Preprocessing Function
# -------------------------------
def preprocess_fn(image):
    # Ensure the input is a NumPy array.
    image = np.array(image)
    
    # Convert to 8-bit if necessary.
    if image.dtype != np.uint8 or image.max() <= 1:
        image = (image * 255).astype(np.uint8)
        
    # Convert from RGB to BGR (OpenCV uses BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to LAB and apply CLAHE on the L-channel
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
    
    # Convert back to RGB
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    return segmented_rgb

# Wrap the preprocessing function for tf.data
def preprocess_tf(images, labels):
    # images: shape (batch_size, 256,256,3)
    def _process(image):
        processed = tf.py_function(func=preprocess_fn, inp=[image], Tout=tf.uint8)
        processed.set_shape([256, 256, 3])
        processed = tf.cast(processed, tf.float32) / 255.0
        return processed
    # Process each image in the batch individually
    processed_images = tf.map_fn(_process, images, dtype=tf.float32)
    return processed_images, labels



# -------------------------------
# 2. Build the FruitVision Model
# -------------------------------
def build_model(input_shape=(256,256,3), num_classes=15):
    inputs = Input(shape=input_shape)
    base_model = MobileNetV3Large(include_top=False, weights='imagenet', input_tensor=inputs)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = build_model(input_shape=(256,256,3), num_classes=15)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# -------------------------------
# 3. Create Datasets using image_dataset_from_directory
# -------------------------------
# Set the data directory. Ensure that the folder contains subdirectories named exactly as in my_classes.
data_dir = './data'

my_classes = [
    'Apple_Fresh',
    'Apple_Rotten',
    'Apple_Formalin-mixed',
    'Banana_Fresh',
    'Banana_Rotten',
    'Banana_Formalin-mixed',
    'Grape_Fresh',
    'Grape_Rotten',
    'Grape_Formalin-mixed',
    'Mango_Fresh',
    'Mango_Rotten',
    'Mango_Formalin-mixed',
    'Orange_Fresh',
    'Orange_Rotten',
    'Orange_Formalin-mixed'
]

# Create training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=my_classes,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=(256,256),
    batch_size=16
)

# Create validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=my_classes,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=(256,256),
    batch_size=16
)

# Apply the custom preprocessing to both datasets
train_ds = train_ds.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)

# Optionally prefetch data for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------
# 4. Training
# -------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('fruitvision_best.h5', monitor='val_loss', save_best_only=True)
]

#CHANGE EPOCHS TO 100 LATER!

history = model.fit(
    train_ds,
    epochs=10, # Increase this value for better results
    validation_data=val_ds,
    callbacks=callbacks
)

# -------------------------------
# 5. Evaluation
# -------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print("Validation Accuracy:", val_acc)