#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fruit Quality Classification - Runner Script
This script serves as a central entry point for running the various scripts in the project.
"""

import os
import sys
import argparse
import subprocess
import shlex

def main():
    parser = argparse.ArgumentParser(description='Fruit Quality Classification Runner')
    parser.add_argument('action', type=str, choices=[
        'train',         # Train the model
        'evaluate',      # Evaluate the model on the full dataset
        'visualize',     # Generate visualizations
        'convert',       # Convert to TFLite and evaluate
        'predict',       # Predict a single image
        'demo',          # Run the demo
    ], help='Action to perform')
    
    # Use remaining arguments after action for the underlying script
    args, unknown_args = parser.parse_known_args()
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define the source directory where scripts are located
    src_dir = os.path.join(project_root, 'src')
    demo_dir = os.path.join(project_root, 'demo')
    models_dir = os.path.join(project_root, 'models')
    
    # Ensure critical directories exist
    for directory in [src_dir, demo_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Map actions to scripts
    scripts = {
        'train': os.path.join(src_dir, 'main-fruit-classification.py'),
        'evaluate': os.path.join(src_dir, 'evaluate_on_full_dataset.py'),
        'visualize': os.path.join(src_dir, 'generate_visualizations.py'),
        'convert': os.path.join(src_dir, 'convert_evaluate_tflite.py'),
        'predict': os.path.join(src_dir, 'predict_image.py'),
        'demo': os.path.join(demo_dir, 'predict.py')
    }
    
    # Add necessary environment variables for model paths
    env = os.environ.copy()
    env['MODELS_DIR'] = models_dir
    
    # Special case for demo
    if args.action == 'demo':
        if not os.path.exists(scripts['demo']):
            print(f"Error: Demo script not found at {scripts['demo']}")
            print("Please check the installation or run setup.py first.")
            return 1
        
        # Make sure demo has access to models
        if not os.path.exists(os.path.join(demo_dir, 'models')):
            print("Creating symlinks to model files for the demo...")
            os.makedirs(os.path.join(demo_dir, 'models'), exist_ok=True)
            
            # Link model files if they don't exist
            for model_file in ['fruit_quality_model.tflite', 'labels.txt', 'class_names.npy']:
                demo_model_path = os.path.join(demo_dir, 'models', model_file)
                source_model_path = os.path.join(models_dir, model_file)
                
                if not os.path.exists(demo_model_path) and os.path.exists(source_model_path):
                    try:
                        # On Unix, create symlink; on Windows copy the file
                        if os.name == 'posix':
                            os.symlink(source_model_path, demo_model_path)
                        else:
                            import shutil
                            shutil.copy2(source_model_path, demo_model_path)
                    except Exception as e:
                        print(f"Warning: Could not link model file {model_file}: {e}")
        
        # Change to the demo directory and run
        os.chdir(demo_dir)
        cmd = [sys.executable, 'predict.py'] + unknown_args
    else:
        # Check if the selected script exists
        script_path = scripts.get(args.action)
        if not script_path or not os.path.exists(script_path):
            print(f"Error: Script for action '{args.action}' not found at {script_path}")
            return 1
        
        # Build the command
        cmd = [sys.executable, script_path] + unknown_args
    
    # Print the command being executed
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except Exception as e:
        print(f"Error executing {args.action}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 