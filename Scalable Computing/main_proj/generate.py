#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image
import re
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=192)
    parser.add_argument('--height', help='Height of captcha image', type=96)
    parser.add_argument(
        '--length', help='Length of captchas in characters', type=int)
    parser.add_argument(
        '--count', help='How many captchas to generate', type=int)
    parser.add_argument(
        '--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument(
        '--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(
        fonts=['fonts/Blastimo.ttf','fonts/Jjester.otf'],
        width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    mapping = []

    for i in range(args.count):
        length = random.randint(1, args.length)
        random_str = ''.join([random.choice(captcha_symbols)
                             for j in range(length)])
        # random_str = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        file_name = f"img_{i:05d}"

        # img.save(os.path.join(output_dir, file_name))
        mapping.append({'filename': file_name + ".png", 'label': random_str})

        image_path = os.path.join(args.output_dir, file_name+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, file_name + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(
                args.output_dir, file_name + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

    # Save mapping
    with open(f'labels_{args.output_dir}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(mapping)


if __name__ == '__main__':
    main()


# python generate.py --width 128 --height 64 --symbols symbols.txt --count 100000 --output-dir train --length 6
