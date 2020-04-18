# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 04_18_2020                                  
# REVISED DATE: 

# Imports python modules
import torch
from torch import nn
import torch.nn.functional as F

### ----------------------------------------------
# Neural network model Classifier class
class Classifier(nn.Module):
    """ Classifier for trained neural network model.
        
        Attributes:
            - Input dimensions: 1024
            - Hidden dimensions: [512, 256]
            - Output dimensions: 102
            
        Args:
            None
            
        Returns:
            Self
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 102) 
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """ Method to do a forward pass through a 
            trained deep learning model.
            
            Args:
                - (torch.tensor) processed image
            Returns:
                - (torch.FloatTensor) LogSoftmax output of the model 
        """
        # Flatten image 
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Return the LogSoftmax of x
        x = F.log_softmax(self.fc3(x), dim=1)

        return x