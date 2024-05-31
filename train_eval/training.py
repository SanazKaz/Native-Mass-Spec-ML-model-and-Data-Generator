from Data_Pre_Processing.Data_loading import load_data, flatten_matrices
from model.model import SpectralNet, get_optimiser, criterion, scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, jaccard_score

# function to calculate the gradient norms
def compute_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, threshold):
    train_epoch_losses = []
    val_epoch_losses = []

    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            X, y = batch
            
            # Forward pass
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # Accumulate batch loss
            train_loss += loss.item()
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            gradient_norm = compute_gradient_norms(model)
            optimizer.step()
        
        scheduler.step()

        # Average loss for the epoch
        train_loss /= len(train_loader)
        train_epoch_losses.append(train_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")        

        # Set model to evaluation mode
        model.eval()
        val_loss = 0.0
        y_true_list = []
        y_pred_list = []  

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                
                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                binary_predictions = (outputs > threshold).float()
                
                y_pred_list.append(binary_predictions.cpu().numpy())
                y_true_list.append(y.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_epoch_losses.append(val_loss)

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        hamming_loss_value = hamming_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        jaccard_score_value = jaccard_score(y_true, y_pred, average='micro')

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Jaccard Score: {jaccard_score_value:.4f}, Hamming Loss: {hamming_loss_value:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }
            torch.save(state, "best_binary_model.pth")
            print("Saved best model to:", "best_binary_model.pth")

    return train_epoch_losses, val_epoch_losses, y_true, y_pred