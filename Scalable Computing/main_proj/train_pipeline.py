#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse
import random

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.2):
    """
    Split dataset into training and validation sets
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        return False
    
    # Create train and val directories
    train_dir = 'train_data'
    val_dir = 'val_data'
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    random.shuffle(image_files)
    
    # Split files
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # Copy files to respective directories
    for filename in train_files:
        shutil.copy2(os.path.join(data_dir, filename), os.path.join(train_dir, filename))
    
    for filename in val_files:
        shutil.copy2(os.path.join(data_dir, filename), os.path.join(val_dir, filename))
    
    print(f"Dataset split complete:")
    print(f"  Training samples: {len(train_files)}")
    print(f"  Validation samples: {len(val_files)}")
    
    return True

def run_command(cmd, description):
    """
    Run a command and handle errors
    """
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete CAPTCHA training pipeline')
    parser.add_argument('--width', help='Width of captcha image', type=int, default=200)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=50)
    parser.add_argument('--max-length', help='Maximum length of captchas', type=int, default=6)
    parser.add_argument('--train-count', help='Number of training captchas to generate', type=int, default=10000)
    parser.add_argument('--val-count', help='Number of validation captchas to generate', type=int, default=2000)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--batch-size', help='Training batch size', type=int, default=32)
    parser.add_argument('--use-gpu', help='Use GPU for training', action='store_true', default=True)
    parser.add_argument('--skip-generation', help='Skip data generation step', action='store_true', default=False)
    parser.add_argument('--skip-training', help='Skip training step', action='store_true', default=False)
    parser.add_argument('--skip-classification', help='Skip classification step', action='store_true', default=False)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAPTCHA Training Pipeline")
    print("=" * 60)
    print(f"Image dimensions: {args.width}x{args.height}")
    print(f"Captcha length: {args.length}")
    print(f"Training samples: {args.train_count}")
    print(f"Validation samples: {args.val_count}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Use GPU: {args.use_gpu}")
    
    # Step 1: Generate training data
    if not args.skip_generation:
        print("\n" + "=" * 40)
        print("STEP 1: Generating Training Data")
        print("=" * 40)
        
        # Generate training data
        train_cmd = [
            'python3', 'enhanced_generate.py',
            '--width', str(args.width),
            '--height', str(args.height),
            '--min-length', '2',
            '--max-length', str(args.max_length),
            '--count', str(args.train_count),
            '--output-dir', 'train_data'
        ]
        
        if not run_command(train_cmd, "Generating training data"):
            print("Failed to generate training data!")
            return
        
        # Generate validation data
        val_cmd = [
            'python3', 'enhanced_generate.py',
            '--width', str(args.width),
            '--height', str(args.height),
            '--min-length', '2',
            '--max-length', str(args.max_length),
            '--count', str(args.val_count),
            '--output-dir', 'val_data'
        ]
        
        if not run_command(val_cmd, "Generating validation data"):
            print("Failed to generate validation data!")
            return
    
    # Step 2: Train the model
    if not args.skip_training:
        print("\n" + "=" * 40)
        print("STEP 2: Training Model")
        print("=" * 40)
        
        train_cmd = [
            'python3', 'variable_length_train.py',
            '--width', str(args.width),
            '--height', str(args.height),
            '--max-length', str(args.max_length),
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--output-model-name', 'captcha_model'
        ]
        
        if args.use_gpu:
            train_cmd.append('--use-gpu')
        
        if not run_command(train_cmd, "Training model"):
            print("Failed to train model!")
            return
    
    # Step 3: Classify test captchas
    if not args.skip_classification:
        print("\n" + "=" * 40)
        print("STEP 3: Classifying Test Captchas")
        print("=" * 40)
        
        classify_cmd = [
            'python3', 'variable_length_classify.py',
            '--model-name', 'captcha_model',
            '--captcha-dir', 'captcha',
            '--output', 'submission.csv',
            '--max-length', str(args.max_length)
        ]
        
        if not run_command(classify_cmd, "Classifying test captchas"):
            print("Failed to classify captchas!")
            return
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("Generated files:")
    print("  - captcha_model.json (model architecture)")
    print("  - captcha_model.h5 (model weights)")
    print("  - submission.csv (predictions)")
    print("\nTo classify new captchas, run:")
    print("  python3 enhanced_classify.py --captcha-dir <your_captcha_dir> --output <output_file>")

if __name__ == '__main__':
    main()
