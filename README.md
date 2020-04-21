# Image Classifier App

Image classification application using Pytorch and Torchvision. This project was developed for Udacity's AI Programming with Python Nanodegree course. The provided code can be used in the following ways:

1. Identify types of flowers using the pretrained netword on 102 flower classes
2. Train a network on your own set of data

The project was split into two tasks:
a. Build and train network to classify flowers in a IPython notebook. 
b. Develop an application that can be run from the command line.

### Frameworks Used
The following application uses Pytorch Torchvision pretrained model from [Densenet-121](https://pytorch.org/docs/stable/torchvision/models.html), and builds a new classifer with 102 classes.   

### Main Files
`train.py` - main code for training and saving model checkpoint
`classifier.py` - classifier architecture
`get_input_args.py` - get command line arguments
`model_functions` - loading data and image transforms
`utility_functions` - image prep

## Training
If you are interested in training your own network you will need to split images into three folders with a similar path as the structure in this code:
    `train`
    `valid`
    `test`

Training can be run from the command line using the example statement with your correct paths. The following receives input arguments from the user: 

`python train.py --data_dir my_folder --save_dir checkpoint.pth --epochs 30 --learning_rate 0.001`

_NOTE:_ The classifier output is 102 classes of flowers. 

Replace `my_folder` with the main directory that your image folders are in. The `--save_dir` can be left alone unless you want to change it. This is where the model checkpoint is saved. Leave the `.pth` extension. Set the `--epochs` and `--learning rate` or accept the default values. 

### CUDA GPU support
The application supports using GPU for training purposes.

## Predict
After you train your network use the `predict.py` app to make infernces on images the network has not seen. Example command line arguments:

`python predict.py --img_path my_img --chk_pt checkpoint.pth --cat_names cat_to_names.json`

`--img_path my_img` - Path to the image you want to classify

`--chk_pt checkpoint.pth` - This is the checkpoint path file location. File must have .pth extension.

`--cat_names cat_to_names.json` - Json file with the 102 categories to flower names, or other class/label combinations. 