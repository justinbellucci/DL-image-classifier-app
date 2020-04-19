# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Training

Example command line statement to run training program. The following receives input arguments from the user: 

`python train.py --data_dir my_folder --save_dir checkpoint.pth --epochs 30 --learning_rate 0.001`

`train.py`
Main training code

`--data_dir my_folder`
This is the main image folder that holds train, valid, and test subfolders. Use exact names for subfolders.

`--save_dir checkpoint.pth`
This is the checkpoint path file location. File must have .pth extension.

`--epochs 30` 
Number of epochs used for training

`--learning_rate 0.001`
Learning rate of optimizer for backpropagation

## Predict

`python predict.py --img_path my_img --chk_pt checkpoint.pth --cat_names cat_to_names.json`

`predict.py`
Main code for predicting the class of an image.

`--img_path my_img`
Path to the image

`--chk_pt checkpoint.pth`
This is the checkpoint path file location. File must have .pth extension.

`--cat_names cat_to_names.json` 
Json file with the 102 categories to flower names. 