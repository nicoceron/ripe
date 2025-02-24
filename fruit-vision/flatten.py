import os
import shutil

def flatten_dataset(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Loop through fruit folders (e.g., "Apple", "Banana", etc.)
    for fruit in os.listdir(source_dir):
        fruit_path = os.path.join(source_dir, fruit)
        if os.path.isdir(fruit_path):
            # Loop through condition folders (e.g., "Fresh", "Rotten", etc.)
            for condition in os.listdir(fruit_path):
                condition_path = os.path.join(fruit_path, condition)
                if os.path.isdir(condition_path):
                    # Create a new folder name by combining fruit and condition
                    new_class_name = f"{fruit}_{condition}"
                    new_class_dir = os.path.join(target_dir, new_class_name)
                    os.makedirs(new_class_dir, exist_ok=True)
                    # Copy all images to the new folder
                    for filename in os.listdir(condition_path):
                        file_path = os.path.join(condition_path, filename)
                        if os.path.isfile(file_path):
                            shutil.copy(file_path, new_class_dir)

# Define source and target directories
source_dir = './data/Augmented-Resized Image'
target_dir = './data/flat_dataset'

flatten_dataset(source_dir, target_dir)
print("Dataset flattened successfully.")