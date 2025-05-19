# Fruit Quality Classification System

A high-accuracy deep learning system for classifying fruit quality and type. This system uses state-of-the-art deep learning techniques including:

- Transfer learning with EfficientNetB3
- Custom model with both CNN and handcrafted features
- Data augmentation for improved robustness
- Advanced color and texture feature extraction
- Fine-tuning strategies
- TFLite model conversion for deployment
- Comprehensive evaluation metrics

## Project Structure

```
.
├── data/                     # Dataset and test images
│   ├── Processed Images_Fruits/  # Training dataset
│   └── test_images/         # Sample test images
├── demo/                    # Demo application
│   ├── models/              # Model files for demo
│   ├── samples/             # Sample images for testing
│   └── predict.py           # Demo script
├── models/                  # Saved models (TF and TFLite)
├── output_files_multiclass/ # Training outputs and results
├── src/                     # Source code
│   ├── main-fruit-classification.py  # Main training script
│   ├── evaluate_on_full_dataset.py   # Evaluation script
│   ├── convert_evaluate_tflite.py    # TFLite conversion
│   ├── generate_visualizations.py    # Generate visualizations
│   ├── improved_training_monitoring.py # Monitoring utilities
│   └── predict_image.py              # Single image prediction
├── run.py                   # Runner script for all functions
└── requirements.txt         # Project dependencies
```

## Dataset Structure

The dataset contains images of various fruits organized by quality:

```
data/Processed Images_Fruits/
├── Good Quality_Fruits/
│   ├── Apple_Good/
│   ├── Banana_Good/
│   ├── Guava_Good/
│   ├── Lime_Good/
│   ├── Orange_Good/
│   └── Pomegranate_Good/
└── Bad Quality_Fruits/
    ├── Apple_Bad/
    ├── Banana_Bad/
    ├── Guava_Bad/
    ├── Lime_Bad/
    ├── Orange_Bad/
    └── Pomegranate_Bad/
```

### Dataset Access

You can access the dataset images from the following sources:

- Mendeley Data: [https://data.mendeley.com/datasets/b6fftwbr2v/1](https://data.mendeley.com/datasets/b6fftwbr2v/1)
- Kaggle: [https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- OpenCV (cv2)

You can install all requirements with:

```bash
pip install -r requirements.txt
```

## How to Use

### Using the Runner Script

The `run.py` script provides a unified interface to all functionality:

```bash
# Train the model
python run.py train

# Evaluate the model on the full dataset
python run.py evaluate

# Generate visualizations
python run.py visualize

# Convert to TFLite and evaluate
python run.py convert

# Predict a single image
python run.py predict path/to/image.jpg

# Run the demo application
python run.py demo
```

### Using the Demo

For a quick demonstration of the model's capabilities:

```bash
# Run the demo with included sample images
cd demo
./run_demo.sh

# Or use the runner script
python run.py demo --list     # List available samples
python run.py demo --sample 1 # Run with sample #1
```

The demo provides a user-friendly way to:

- Test the model with sample images
- Upload and analyze your own fruit images
- View detailed prediction results
- Get quality classification with confidence scores

See the `demo/README.md` for more details.

### Direct Script Usage

You can also run each script directly from the `src` directory:

#### Training the Model

```bash
python src/main-fruit-classification.py
```

The script will:

1. Analyze your dataset structure
2. Create appropriate data loaders with augmentation
3. Train an EfficientNetB3-based model with transfer learning
4. Fine-tune the model for optimal performance
5. Save the best model automatically
6. Create detailed performance visualizations

Training outputs will be saved to the `output_files_multiclass/` directory.

#### Evaluating the Model

```bash
python src/evaluate_on_full_dataset.py
```

#### Converting to TFLite

```bash
python src/convert_evaluate_tflite.py
```

#### Generating Visualizations

```bash
python src/generate_visualizations.py
```

#### Predicting a Single Image

```bash
python src/predict_image.py path/to/image.jpg
```

## Model Architecture

The system uses a sophisticated dual-input architecture:

1. **Image Processing Branch**:

   - EfficientNetB3 pre-trained on ImageNet
   - Data augmentation
   - Global Average Pooling

2. **Handcrafted Features Branch**:

   - 117 color, texture, and shape features
   - Dense layer with batch normalization

3. **Combined Processing**:
   - Feature concatenation
   - Multiple dense layers with swish activation
   - Batch normalization and dropout for regularization
   - Softmax output layer

## Training Strategy

The training process uses a two-phase approach:

1. **Initial Phase**:

   - Frozen base model
   - Training only the top layers
   - Higher learning rate

2. **Fine-tuning Phase**:
   - Unfreezing top layers of the base model
   - Lower learning rate
   - Continue training

## Performance Optimization

- Mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Batch normalization
- Focal loss for handling class imbalance

## TFLite Conversion

For deployment on edge devices, the model can be converted to TFLite format with the conversion script. The TFLite model maintains high accuracy while being optimized for mobile and edge devices.

## Troubleshooting

If you encounter issues:

1. Check TensorFlow version compatibility
2. Ensure dataset is structured correctly
3. Consider using a GPU for faster training
4. Try with a smaller batch size if experiencing memory issues
