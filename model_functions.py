# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 04_18_2020                                  
# REVISED DATE: 04_19_2020

# Imports python modules
import torch
from torchvision import datasets, transforms, models
from classifier import Classifier

# Define transforms and load datasets
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 32
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    # Define transforms for the training, validation, and testing sets 
    train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(img_mean, img_std)])

    valid_test_transforms = transforms.Compose([transforms.Resize(226),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(img_mean, img_std)])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)
    
    return train_loader, valid_loader, test_loader, train_data

    
### ----------------------------------------------
# Save model checkpoint
def save_checkpoint(train_data, epochs, model, path):
    """ Function to save the model checkpoint
        Checkpoint keys:
            - epochs (epoch number)
            - state_dict (model.state_dict())
            -class_to_idx (model.class_to_idx())
        
        Args:
            - epochs (int) 
            - model  
            - path (str ex. "/checkpoint.pth"
        
        Returns:
            - Model checkpoint file (extension .pth)
    """
    # move model to cpu to avoid loading issues
    device = torch.device("cpu") 
    model.to(device) 
    
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    return torch.save(checkpoint, path)

### ----------------------------------------------
def load_checkpoint(filepath):
    """Load a model checkpoint and rebuild model architecture
    
       Arguments:
       - filepath (path to checkpoint.pth file)
       
       Output:
       - model (trained deep learning model)
       - class_dict (class_to_idx dictionary)
    
    """
    model = models.densenet121(pretrained=True)
    model.classifier = Classifier()
#     checkpoint = torch.load(filepath, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
    