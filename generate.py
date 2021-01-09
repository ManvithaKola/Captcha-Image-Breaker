#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image
import numpy as np
import codecs

#encode
def encode_img_name(img_name):
    hex_string = codecs.encode(img_name.encode(), "hex")
    hex_string = hex_string.decode("ASCII")
    return hex_string
'''
#decode
def decode_img_name(hex_img_name):
    bytes_object = bytes.fromhex(hex_img_name)
    ascii_string = bytes_object.decode("ASCII")
    return ascii_string

#rotates an image by given angle
#Ref:https://stackoverflow.com/a/9042907/1761743
def transform_image(image):
  pts_to = np.float32([[0,0],[image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
  tr_range = 5
  rand_num = np.float32([[random.randint(0,tr_range), random.randint(0,tr_range)],
              [-1 *random.randint(0,tr_range),  random.randint(0,tr_range)],
              [random.randint(0,tr_range), -1 * random.randint(0,tr_range)],
              [-1 * random.randint(0,tr_range), -1 * random.randint(0,tr_range)]])
  pts_from = pts_to + rand_num
  img_trans = cv2.getPerspectiveTransform(pts_from, pts_to)
  result = cv2.warpPerspective(image, img_trans, (image.shape[1], image.shape[0]))
  return result
'''
def get_image_path(img_dir, random_str):
    random_str = encode_img_name(random_str)
    image_path = os.path.join(img_dir, random_str + '.png')
    if os.path.exists(image_path):
        version = 1
        while os.path.exists(os.path.join(img_dir, random_str + '_' + str(version) + '.png')):
            version += 1
        image_path = os.path.join(img_dir, random_str + '_' + str(version) + '.png')
    return image_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
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

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for i in range(args.count):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        image_path = get_image_path(args.output_dir, random_str)
        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)


if __name__ == '__main__':
    main()
