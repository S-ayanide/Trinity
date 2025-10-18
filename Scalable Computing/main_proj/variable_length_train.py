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
from collections import Counter

# Build a Keras model for variable length captchas
def create_variable_length_model(max_captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    
    # Convolutional layers
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    
    # Create outputs for each possible position (up to max_captcha_length)
    outputs = []
    for i in range(max_captcha_length):
        outputs.append(keras.layers.Dense(captcha_num_symbols, activation='softmax', name=f'char_{i+1}')(x))
    
    model = keras.Model(inputs=input_tensor, outputs=outputs)
    return model

# Sequence class for variable length captchas
class VariableLengthImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, max_captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.max_captcha_length = max_captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)
        
        # Analyze length distribution
        lengths = [len(filename.split('.')[0]) for filename in file_list]
        self.length_distribution = Counter(lengths)
        print(f"Length distribution in dataset: {dict(self.length_distribution)}")

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.max_captcha_length)]

        for i in range(self.batch_size):
            # Simple approach: just pick a random file from all files, no complex management
            all_file_labels = list(self.files.keys())
            if not all_file_labels:
                # If files dict is empty, rebuild it from used_files
                self.files = {label: label + '.png' for label in self.used_files}
                self.used_files = []
                all_file_labels = list(self.files.keys())
            
            random_image_label = random.choice(all_file_labels)
            random_image_file = self.files[random_image_label]

            # Load and preprocess image
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            
            # Resize image to expected dimensions if needed
            if processed_data.shape[:2] != (self.captcha_height, self.captcha_width):
                processed_data = cv2.resize(processed_data, (self.captcha_width, self.captcha_height))
            
            X[i] = processed_data

            # Handle variable length labels
            random_image_label = random_image_label.split('_')[0]
            actual_length = len(random_image_label)
            
            # Set labels for actual characters
            for j, ch in enumerate(random_image_label):
                if j < self.max_captcha_length:
                    y[j][i, :] = 0
                    y[j][i, self.captcha_symbols.find(ch)] = 1
            
            # For positions beyond actual length, set to a special "end" token or padding
            # We'll use the last character as a padding token
            padding_token_idx = len(self.captcha_symbols) - 1  # Use last symbol as padding
            for j in range(actual_length, self.max_captcha_length):
                y[j][i, :] = 0
                y[j][i, padding_token_idx] = 1

        # Convert list to tuple for proper output format
        return X, tuple(y)

def setup_gpu():
    """
    Setup GPU configuration for training
    """
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(f"Found {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            return '/device:GPU:0'
        else:
            print("No GPU found, using CPU")
            return '/device:CPU:0'
    except Exception as e:
        print(f"GPU setup failed: {e}, using CPU")
        return '/device:CPU:0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int, default=200)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=50)
    parser.add_argument('--max-length', help='Maximum length of captchas', type=int, default=6)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int, default=32)
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str, default='train_data')
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str, default='captcha')
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str, default='captcha_model')
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int, default=50)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str, default='sample-code/symbols.txt')
    parser.add_argument('--use-gpu', help='Use GPU for training', action='store_true', default=True)
    args = parser.parse_args()

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print(f"Training with symbol set: {captcha_symbols}")
    print(f"Number of symbols: {len(captcha_symbols)}")
    print(f"Maximum captcha length: {args.max_length}")
    print(f"Image dimensions: {args.width}x{args.height}")

    # Setup device (GPU or CPU)
    device = setup_gpu() if args.use_gpu else '/device:CPU:0'
    print(f"Using device: {device}")

    with tf.device(device):
        model = create_variable_length_model(args.max_length, len(captcha_symbols), (args.height, args.width, 3))

        if args.input_model is not None:
            print(f"Loading existing model from {args.input_model}")
            model.load_weights(args.input_model)

        # Create loss and metrics for each output
        losses = ['categorical_crossentropy'] * args.max_length
        metrics = ['accuracy'] * args.max_length
        
        model.compile(loss=losses,
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=metrics)

        model.summary()

        training_data = VariableLengthImageSequence(args.train_dataset, args.batch_size, args.max_length, captcha_symbols, args.width, args.height)
        validation_data = VariableLengthImageSequence(args.validate_dataset, args.batch_size, args.max_length, captcha_symbols, args.width, args.height)

        callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=True, monitor='val_loss')]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        print(f"Starting training for {args.epochs} epochs...")
        print(f"Training samples: {training_data.count}")
        print(f"Validation samples: {validation_data.count}")

        try:
            history = model.fit(training_data,
                                validation_data=validation_data,
                                epochs=args.epochs,
                                callbacks=callbacks)
            
            print("Training completed successfully!")
            
            # Save final model
            model.save_weights(args.output_model_name+'_final.h5')
            print(f"Final model saved as {args.output_model_name}_final.h5")
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')

if __name__ == '__main__':
    main()
