# CNN based Captcha Image Breaker using tflite

## Requirements
python-captcha, opencv_python, python-tensorflow (CPU or GPU), tflite, tflite_runtime

## Generating captchas
```
python generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 30000 --output-dir training_data
python generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 10000 --output-dir validation_data
python generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 1000 --output-dir captcha_dir
```
This generates 30000 image captchas with 4 characters per captcha, using the set of symbols in the `symbols.txt` file. The captchas are stored in the folder
`training_data`, which is created if it doesn't exist. The same with the validation and the test(captcha_dir) data sets

## Training the neural network

```
python train.py --width 128 --height 64 --length 4 --symbols symbols.txt --batch-size 32 --epochs 10 --train-dataset training_data --validate-dataset validation_data
```

Train the neural network for 10 epochs on the generated data . One epoch is one pass through the entire dataset.
The suggested training dataset size for the initial training for captcha length of 4 symbols is 30000 images, with a validation dataset size of 10000 images.
The tensorflow model is then converted to tensorflow lite model and the converted model is saved into converted_model.tflite file

## Running the classifier

```
python classify.py --model-name model --modellite converted_model --captcha-dir captcha_dir --output stuff.txt --outputlite stufflite.txt --symbols symbols.txt
```

With `--model-name model` the classifier python code will look for a model called `model.json` with weights `model.h5` in the current directory, and load the model.
With `--modellite converted_model` the classifier python code will look for a tflite model called `converted_model.json` in the current directory, and load the tflite model.
The classifier code runs through all the images in `--captcha-dir` through the model, and saves the file names and the model's captcha prediction in the `--output` file.


