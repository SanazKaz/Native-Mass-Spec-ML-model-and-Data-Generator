import torch
import numpy as np
import os
from Mass_Spec_ML_Project.Final_Structure.model import get_model, get_optimiser, criterion, scheduler, load_checkpoint
from torch.utils.data import DataLoader, TensorDataset
from Mass_Spec_ML_Project.Final_Structure.Data_Pre_Processing.Data_loading import load_data as ld, flatten_matrices as fm
from Mass_Spec_ML_Project.Final_Structure.Data_Pre_Processing.Data_splitting import pred_tensordataset as ptd
from Mass_Spec_ML_Project.Final_Structure.train_eval.eval import evaluate


def predict(path_x, path_y, batch_size, lr, max_lr, filename='checkpoint.pth.tar'):
    # Load the data
    x = ld(path_x)
    y = ld(path_y)
    y = fm(y) # flatten the matrices

    # create a dataloader from it
    loader = ptd(x, y, batch_size=batch_size) 

    # Load the model
    model = get_model()
    optimizer = get_optimiser(model, lr=lr)
    scheduler = scheduler(optimizer, base_lr=lr, max_lr=max_lr, step_size_up=2000, mode='triangular2')

    # Load the checkpoint
    model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, criterion(), filename=filename)

    # Predict
    y_true, y_pred, X_true = evaluate(model, loader, criterion())
    
    return y_true, y_pred, X_true


if __name__ == "__main__":
    y_true, y_pred, X_true = predict("path/to/x.pkl", "path/to/y.pkl", batch_size=32, lr=0.001, max_lr=0.01)
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"X_true shape: {X_true.shape}")