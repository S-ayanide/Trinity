#!/usr/bin/env python3
"""
Script to prepare training/validation split and train the model
"""

import os
import shutil
import random
from pathlib import Path

def prepare_datasets(source_dir, train_dir, val_dir, val_split=0.2):
    """Split the dataset into training and validation sets"""
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all PNG files
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    random.shuffle(all_files)
    
    # Calculate split
    val_count = int(len(all_files) * val_split)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Copy files to respective directories
    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    for f in val_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(val_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Directories
    source_dir = "test"
    train_dir = "train_data"
    val_dir = "val_data"
    
    # Prepare datasets
    prepare_datasets(source_dir, train_dir, val_dir, val_split=0.15)
    
    print("\nNow training the model...")
    print("=" * 60)
    
    # Train the model with better parameters
    os.system("""python3 train.py \
        --width 128 \
        --height 64 \
        --length 5 \
        --batch-size 64 \
        --epochs 110 \
        --train-dataset train_data \
        --validate-dataset val_data \
        --output-model-name captcha_model \
        --symbols symbols.txt""")

