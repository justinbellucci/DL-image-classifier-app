# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 04_18_2020                                  
# REVISED DATE: 

# Imports python modules
import torch
# import torch.nn.functional as F
from torch import nn, optim
# from collections import OrderedDict
from torchvision import models
from workspace_utils import active_session
# from PIL import Image

import model_functions 
from classifier import Classifier
from get_input_args import get_input_args

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import json

def main():
    
    # Get command line input arguments
    in_arg = get_input_args()
    
    # Load data
    train_loader, valid_loader, test_loader, train_data = model_functions.load_data(in_arg.data_dir)
    
    # Define Learning Rate
    learn_rate = in_arg.learning_rate

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True) # Download the model

    # Freeze parameters in model feature so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Create an instance of the Classifier object 
    model.classifier = Classifier()

    # Create an instance of the Loss object
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    model.to(device) # Move model to either CPU or GPU

    train_losses, valid_losses, test_losses = [], [], []
    epochs = in_arg.epochs

    with active_session():
        for e in range(epochs):
            running_loss = 0
            # ----------- Training Pass ----------- 
            for images, labels in train_loader:
                # Move image and label tensors to default device
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad() # Zero gradient
                log_pbs = model.forward(images) # Forward pass  
                loss = criterion(log_pbs, labels) # Calulate the loss
                loss.backward() # Calculate gradients with loss
                optimizer.step() # Update weights
                running_loss += loss.item()  

                # ---------- Validation Pass ----------
            else:
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval() # Eval mode - dropout off
                    for images, labels in valid_loader:
                        # Move image and label tensors to default device
                        images, labels = images.to(device), labels.to(device)
                        valid_log_pbs = model.forward(images) # Forward pass
                        valid_loss += criterion(valid_log_pbs, labels) # Calculate validation loss

                        pbs = torch.exp(valid_log_pbs) # Calculate validation probabilities
                        top_p, top_c = pbs.topk(1, dim=1) # Get top predicted class from our model
                        equality = top_c == labels.view(*top_c.shape) # Check if top class == label
                        accuracy += torch.mean(equality.type(torch.FloatTensor)) # Calculate accuracy 

                model.train() # Train mode
                train_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))

                # --- Print Statements ---
                print("Epochs: {}/{} -- ".format(e+1, epochs),
                      "Training Loss: {:.3f} -- ".format(running_loss/len(train_loader)),
                      "Valid Loss: {:.3f} -- ".format(valid_loss/len(valid_loader)),
                      "Valid Accuracy: {:.3f}% -- ".format(accuracy.item()/len(valid_loader)*100))
  
    ## -------------
    # Run model on test images to get accuracy on new images
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # Transfer Tensors to default device
            test_log_pbs = model.forward(images) # Forward pass
            test_loss += criterion(test_log_pbs, labels) # Calculate loss

            ps = torch.exp(test_log_pbs)
            top_p, top_c = ps.topk(1, dim=1)
            equality = top_c == labels.view(*top_c.shape)
            test_accuracy += torch.mean(equality.type(torch.FloatTensor))

    model.train() # Train mode
    test_losses.append(test_loss/len(test_loader))

    # --- Print Statements ---
    print("Test Loss: {:.3f} -- ".format(test_loss/len(test_loader)),
          "Test Accuracy: {:.3f}% -- ".format(test_accuracy.item()/len(test_loader)*100))

        ## -------------
    # Save the model checkpoint
    model_functions.save_checkpoint(train_data, epochs, model, in_arg.save_dir)
    
    # Call to main function to run the program
if __name__ == "__main__":
    main()