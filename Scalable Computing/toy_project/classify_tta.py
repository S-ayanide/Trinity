#!/usr/bin/env python3
"""
Classification with Test-Time Augmentation for better accuracy
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from collections import Counter

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def augment_image(image, aug_type):
    """Apply slight augmentation for TTA"""
    if aug_type == 0:
        return image  # Original
    elif aug_type == 1:
        # Slight brightness increase
        return numpy.clip(image * 1.05, 0, 1)
    elif aug_type == 2:
        # Slight brightness decrease
        return numpy.clip(image * 0.95, 0, 1)
    elif aug_type == 3:
        # Add tiny noise
        noise = numpy.random.normal(0, 0.01, image.shape)
        return numpy.clip(image + noise, 0, 1)
    elif aug_type == 4:
        # Slight contrast
        mean = numpy.mean(image)
        return numpy.clip((image - mean) * 1.1 + mean, 0, 1)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--tta-rounds', help='Number of TTA rounds (default 5)', type=int, default=5)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with TTA (rounds=" + str(args.tta_rounds) + ")")

    with tf.device('/cpu:0'):
        json_file = open(args.model_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(args.model_name+'.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        all_files = [f for f in os.listdir(args.captcha_dir) 
                     if not f.startswith('.') and f.endswith('.png')]
        all_files.sort()
        
        results = []
        for x in all_files:
            # Load and preprocess image
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (h, w, c) = image.shape
            
            # Perform multiple predictions with augmentations
            predictions = []
            for aug_id in range(args.tta_rounds):
                aug_image = augment_image(image.copy(), aug_id)
                aug_image = aug_image.reshape([-1, h, w, c])
                prediction = model.predict(aug_image, verbose=0)
                captcha_text = decode(captcha_symbols, prediction)
                predictions.append(captcha_text)
            
            # Vote for each character position
            captcha_length = len(predictions[0])
            final_captcha = []
            for char_pos in range(captcha_length):
                chars_at_pos = [pred[char_pos] for pred in predictions]
                # Most common character at this position
                most_common = Counter(chars_at_pos).most_common(1)[0][0]
                final_captcha.append(most_common)
            
            final_text = ''.join(final_captcha)
            results.append((x, final_text))
            print(f'Classified {x} -> {final_text} (votes: {predictions})')
        
        # Write results
        with open(args.output, 'w') as output_file:
            for filename, captcha in results:
                output_file.write(filename + "," + captcha + "\n")
        
        print(f"\n✓ Results saved to {args.output}")

if __name__ == '__main__':
    main()

