# Fruit Quality Classification Demo

This demo allows you to test the fruit quality classification model on your own images or the provided sample images.

## Directory Structure

```
demo/
├── models/             # Model files
│   ├── fruit_quality_model.tflite    # TensorFlow Lite model
│   ├── labels.txt                   # Class labels
│   └── class_names.npy              # Class names in numpy format
├── samples/            # Sample images for testing
├── results/            # Results will be saved here (auto-created)
├── predict.py          # Main prediction script
└── README.md           # This file
```

## Requirements

- Python 3.6+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

Install requirements from the root directory:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# List available sample images
python predict.py --list

# Run prediction on a specific sample (by index)
python predict.py --sample 1

# Run prediction on a custom image
python predict.py --image_path /path/to/your/image.jpg

# Specify custom model or labels file (optional)
python predict.py --model /path/to/your/model.tflite --labels /path/to/your/labels.txt
```

## Sample Output

The prediction results will show:

1. The input image
2. A bar chart with the top 3 predictions and their probabilities
3. Text output of the top 3 predictions

Results are also saved to the `results/` directory.

## Model Information

This model classifies fruits into 12 categories combining fruit type and quality:

- Apple (Good/Bad)
- Banana (Good/Bad)
- Guava (Good/Bad)
- Lime (Good/Bad)
- Orange (Good/Bad)
- Pomegranate (Good/Bad)

The model takes two inputs:

1. Image (224x224 RGB image)
2. Handcrafted features (color histograms, texture, and shape features)

And outputs classification probabilities for each class.
