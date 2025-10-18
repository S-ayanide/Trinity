#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

def decode_variable_length(characters, predictions, max_length):
    """
    Decode variable length predictions by finding the end token
    """
    decoded_text = ""
    padding_token_idx = len(characters) - 1  # Last symbol is padding token
    
    for i in range(max_length):
        char_pred = numpy.argmax(predictions[i][0])
        confidence = numpy.max(predictions[i][0])
        
        # Don't use padding token at all - just decode all positions
        # and let the user decide what's valid
        if char_pred != padding_token_idx:
            decoded_text += characters[char_pred]
        else:
            # If it's a padding token, try the second best prediction
            sorted_indices = numpy.argsort(predictions[i][0])[::-1]
            second_best_idx = sorted_indices[1]
            second_best_confidence = predictions[i][0][second_best_idx]
            
            # Use second best if it has reasonable confidence
            if second_best_confidence > 0.1:
                decoded_text += characters[second_best_idx]
            else:
                # If no good alternative, just skip this position
                pass
    
    return decoded_text

def setup_device():
    """
    Setup device for inference (prefer CPU for 32-bit compatibility)
    """
    try:
        # For 32-bit Raspberry Pi compatibility, prefer CPU
        return '/device:CPU:0'
    except Exception as e:
        print(f"Device setup failed: {e}, using default")
        return '/device:CPU:0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str, default='captcha_model')
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str, default='captcha')
    parser.add_argument('--output', help='File where the classifications should be saved', type=str, default='submission.csv')
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str, default='sample-code/symbols.txt')
    parser.add_argument('--max-length', help='Maximum captcha length', type=int, default=6)
    args = parser.parse_args()

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    print(f"Model: {args.model_name}")
    print(f"Input directory: {args.captcha_dir}")
    print(f"Output file: {args.output}")
    print(f"Maximum length: {args.max_length}")

    device = setup_device()
    print(f"Using device: {device}")

    with tf.device(device):
        with open(args.output, 'w') as output_file:
            # Load model architecture
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            
            # Load model weights
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            # Get list of captcha files
            captcha_files = [f for f in os.listdir(args.captcha_dir) if f.endswith('.png')]
            print(f"Found {len(captcha_files)} captcha images to classify")

            # Process each captcha
            for i, filename in enumerate(captcha_files):
                try:
                    # Load image and preprocess it
                    raw_data = cv2.imread(os.path.join(args.captcha_dir, filename))
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                    image = numpy.array(rgb_data) / 255.0
                    
                    # Resize image to match model's expected input dimensions (50x200)
                    image = cv2.resize(image, (200, 50))
                    
                    # Reshape for model input
                    image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
                    
                    # Make prediction
                    predictions = model.predict(image, verbose=0)
                    
                    # Decode prediction
                    decoded_text = decode_variable_length(captcha_symbols, predictions, args.max_length)
                    
                    # Write to output file
                    output_file.write(filename + ", " + decoded_text + "\n")
                    
                    if (i + 1) % 100 == 0:
                        print(f'Classified {i + 1}/{len(captcha_files)} captchas...')
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    # Write empty prediction for failed cases
                    output_file.write(filename + ", \n")
                    continue

            print(f'Classification complete! Results saved to {args.output}')

if __name__ == '__main__':
    main()
