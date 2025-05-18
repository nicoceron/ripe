#!/bin/bash

# Fruit Quality Classification Demo Runner Script

# Change to the script directory
cd "$(dirname "$0")"

# Check if Python 3 is available
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

# Check if required packages are installed
$PYTHON -c "import tensorflow, cv2, numpy, matplotlib" &>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r ../requirements.txt
fi

# Check if samples directory has images
sample_count=$($PYTHON -c "from pathlib import Path; print(len(list(Path('samples').glob('*.jpg')) + list(Path('samples').glob('*.png')) + list(Path('samples').glob('*.jpeg'))))")
if [ "$sample_count" -eq 0 ]; then
    echo "No sample images found. Using --image_path instead."
    
    # Check if there are test images in the data directory
    if [ -d "../data/test_images" ]; then
        test_image=$(find ../data/test_images -type f -name "*.jpg" -o -name "*.png" | head -n 1)
        if [ -n "$test_image" ]; then
            echo "Running with test image: $test_image"
            $PYTHON predict.py --image_path "$test_image"
            exit 0
        fi
    fi
    
    echo "Error: No sample images available. Please add images to the samples directory."
    exit 1
fi

# List sample images
echo "Available sample images:"
$PYTHON predict.py --list

# Run with first sample image
echo -e "\nRunning prediction on first sample image...\n"
$PYTHON predict.py 