import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from .Data_loading import load_data, flatten_matrices

logging.basicConfig(level=logging.INFO)

def add_noise(x_test, noise_level):
    """Adds noise to spectra 
    args:
    x_test: list of test spectra
    noise_level: standard deviation of the noise to be added"""
    
    noisy_x_test = []
    for i, x_test in enumerate(x_test):
        noise = np.random.normal(0, noise_level, size=(2001))
        noisy_x_test[i] = x_test + noise
        noisy_x_test.append(noisy_x_test[i])

    return noisy_x_test


def train_val_test_split(x, y): # x is spectra, y is interaction matrices   
    """split your data into training, validation and test sets"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test

def create_tensordatasets(x_train, x_val, x_test, y_train, y_val, y_test):
    """create pytorch tensordatasets"""
    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)
    noisy_test_set = TensorDataset(x_test, y_test) # need to add noise to x_test in here
    return train_set, val_set, test_set, noisy_test_set

def create_data_loaders(train_set, val_set, test_set, noisy_test_set, batch_size):
    """create pytorch data loaders"""
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    noisy_test_loader = DataLoader(noisy_test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, noisy_test_loader
    
def pred_tensordataset(x, y, batch_size):
    """create pytorch tensordatasets"""
    set = TensorDataset(x, y)
    set_loaded = DataLoader(set, batch_size=batch_size, shuffle=False)
    return set_loaded



if __name__ == "__main__":
    # Load the data using the functions from Data_loading.py
    x_data = load_data("/Users/sanazkazeminia/Documents/Mass_Spec_project/Mass_Spec_ML_Project/4x4_spectra_dataset_10binned.pkl")
    y_data = load_data("/Users/sanazkazeminia/Documents/Mass_Spec_project/Mass_Spec_ML_Project/4x4_interaction_matrices_10binned.pkl")
    flattened_y = flatten_matrices(y_data)

    # Split the data into train, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x_data, flattened_y)

    # Create TensorDatasets and DataLoaders
    train_set, val_set, test_set, noisy_test_set = create_tensordatasets(x_train, x_val, x_test, y_train, y_val, y_test)
    train_loader, val_loader, test_loader, noisy_test_loader = create_data_loaders(train_set, val_set, test_set, noisy_test_set, batch_size=32)

    # Print the shapes of the split data
    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")


logging.info("Script executed successfully")
