import pickle as pkl
import torch
import torch
import logging

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

# need to flatten the matrices for the model

def flatten_matrices(Interaction_matrices):
    for matrix in Interaction_matrices: # forgot to zero out 0,0 
        matrix[0,0] = 0
    flattened_matrices = [matrix.flatten() for matrix in Interaction_matrices] # storing in an array
    flattened_matrices = torch.stack(flattened_matrices) # stacking the array
    return flattened_matrices


# Log the script
logging.info("Script executed successfully")

if __name__ == "__main__":
    x_data = load_data("path/to/x.pkl") # replace with actual path
    y_data = load_data("path/to/y.pkl") # replace with actual path
    flattened_y = flatten_matrices(y_data)
    print(f"x_data shape: {x_data.shape}")
    print(f"flattened_y shape: {flattened_y.shape}")