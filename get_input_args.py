# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 04_18_2020                                  
# REVISED DATE: 

# Imports python modules
import argparse

def get_input_args():
    """ Function that retrieves the following 3 command line inputs 
        from the user using the Argparse Python module. If the user fails to 
        provide some or all of the 3 inputs, then the default values are
        used for the missing inputs. Command Line Arguments:
    
        Command Line Arguments:
            1. --data_dir (str) Image folder that has train, valid, test folders in it
            2. --save_dir (str) Directory where model checkpoint gets saved
            3. --epochs (int) Number of epochs to train  
            4. --learning_rate (float) Learning rate for optimizer
            
        Parameters:
            None
        
        Returns:
            parse_args() Data structure that stores the command line arguemnts object
    """   
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()   
    # Argument 1: Data directory path
    parser.add_argument('--data_dir', type = str, default = './flowers', help = 'Path to master images folder')    
    # Argument 2: Checkpoint data directory path
    parser.add_argument('--save_dir', type = str, default = './', help = 'Path to save model checkpoint')    
    # Argument 3: Number of epochs 
    parser.add_argument('--epochs', type = int, default = 35, help = 'Epochs (int) for training')    
    # Argument 3: Number of epochs 
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate')    
    # Assign variable in_args to parse_args()
    in_args = parser.parse_args()
    
    return in_args