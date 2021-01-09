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

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',default='model', help='Model name to use for classification', type=str)
    parser.add_argument('--modellite',default='converted_model', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir',default='captcha_dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output',default='stuff.txt', help='File where the classifications should be saved', type=str)
    parser.add_argument('--outputlite',default='stufflite.txt', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols',default='symbols.txt', help='File with the symbols to use in captchas', type=str)
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

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/gpu:0'):
        with open(args.output, 'w',newline='\n') as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-4, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                s = decode(captcha_symbols, prediction)
                output_file.write(x + "," + s.replace(" ", "") + "\n")

                print('Classified using TF ' + x)
        print("==========================================================================================")
        with open(args.outputlite, 'w',newline='\n') as output_file:
            interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
            interpreter.allocate_tensors()
            for x in os.listdir(args.captcha_dir):
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w]).astype(numpy.float32)
                input_index = interpreter.get_input_details()[0]["index"]
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_index, image)
                interpreter.invoke()
                prediction = []
                for i in range(4):
                    prediction.append(numpy.array(interpreter.get_tensor(output_details[i]['index'])))
                s = decode(captcha_symbols, prediction)
                output_file.write(x + "," + s.replace(" ", "") + "\n")
                
                print('Classified using TFLite' + x)

if __name__ == '__main__':
    main()
