# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 04_19_2020                                  
# REVISED DATE: 

# Imports python modules
import torch
from PIL import Image
import json
import numpy as np

def process_image(image):
    """ Scales, crops (224 x 224px), and normalizes a PIL image 
        for a PyTorch model, returns an Pytorch Tensor
        
        Arguments:
        - jpg image
        
        Output:
        - Pytorch Tensor (image)
    """
    dim = 224
    # --- Resize image
    im = Image.open(image)
    width, height = im.size
    if width > height:
        ratio = width/height
        im.thumbnail((ratio*256, 256))
    elif height > width:
        ratio = height/width
        im.thumbnail((256, ratio*256))
    new_width, new_height = im.size

    # --- Crop image around center
    left = (new_width - dim)/2
    top = (new_height - dim)/2
    right = (new_width + dim)/2
    bottom = (new_height + dim)/2
    im = im.crop((left, top, right, bottom))

    # Convert to np.array and divide by color channel int max
    np_image = np.array(im)/255

    # Normalize color channels
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (np_image - mean)/std

    # Convert to a Tensor
    image = torch.FloatTensor(image.transpose(2, 0, 1)) 
    
    return image

def import_cat_to_names(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name