import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import wandb
import matplotlib.pyplot as plt


def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    X_true = []
    

    with torch.no_grad():
        for batch in loader:
            X, y = batch
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            binary_predictions = (outputs > 0.5).float()

            y_pred.extend(binary_predictions.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            X_true.extend(X.cpu().numpy())
    val_loss /= len(loader)
    
    print(f"Loss: {val_loss:.4f}")

    return y_true, y_pred, X_true, val_loss



import numpy as np

def label_based_macro_precision(y_true, y_pred):
    # Exclude the first class/label (position 0)
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    
    # axis = 0 computes true positive along columns i.e., labels
    l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis=0)
    
    # axis = 0 computes true_positive + false positive along columns i.e., labels
    l_prec_den = np.sum(y_pred, axis=0)
    
    # compute precision per class/label
    l_prec_per_class = l_prec_num / l_prec_den
    
    # macro precision = average of precision across labels
    l_prec = np.mean(l_prec_per_class)
    
    return l_prec




def label_based_macro_accuracy(y_true, y_pred):

    # Exclude the first class/label (position 0)
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
	
    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = np.sum(np.logical_or(y_true, y_pred), axis = 0)

    # compute mean accuracy across labels. 
    return np.mean(l_acc_num/l_acc_den)


def label_based_macro_recall(y_true, y_pred):

    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    
    # compute true positive along axis = 0 i.e labels
    l_recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = np.sum(y_true, axis = 0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num/l_recall_den

    # compute macro averaged recall i.e recall averaged across labels. 
    l_recall = np.mean(l_recall_per_class)
    return l_recall

def label_based_micro_accuracy(y_true, y_pred):

    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    
    # sum of all true positives across all examples and labels 
    l_acc_num = np.sum(np.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = np.sum(np.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num/l_acc_den



def label_based_micro_precision(y_true, y_pred):

    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    
    # compute sum of true positives (tp) across training examples
    # and labels. 
    l_prec_num = np.sum(np.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num/l_prec_den

# Function for Computing Label Based Micro Averaged Recall 
# for a MultiLabel Classification problem. 

def label_based_micro_recall(y_true, y_pred):
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
	
    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_true)

    # compute mirco-average recall
    return l_recall_num/l_recall_den

def alpha_evaluation_score(y_true, y_pred):

    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]

    alpha = 1
    beta = 0.25
    gamma = 1
    
    # compute true positives across training examples and labels
    tp = np.sum(np.logical_and(y_true, y_pred))
    
    # compute false negatives (Missed Labels) across training examples and labels
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    
    # compute False Positive across training examples and labels.
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
        
    # Compute alpha evaluation score
    alpha_score = (1 - ((beta * fn + gamma * fp ) / (tp +fn + fp + 0.00001)))**alpha 
    
    return alpha_score


def hamming_loss(y_true, y_pred):

    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den


def jaccard_score(y_true, y_pred):

    # compute intersection between true and predicted labels
    intersection = np.sum(np.logical_and(y_true, y_pred))
    
    # compute union between true and predicted labels
    union = np.sum(np.logical_or(y_true, y_pred))
    
    # compute jaccard score
    return intersection/union


def plot_metrics(y_true, y_pred):
    # Calculate per-label metrics
    per_label_precision = precision_score(y_true, y_pred, average="macro")
    per_label_recall = recall_score(y_true, y_pred, average="macro")
    per_label_f1 = f1_score(y_true, y_pred, average="macro")
    
    # Create labels for x-axis
    labels = [f'{i+1}' for i in range(y_true.shape[1])]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot precision
    axs[0, 0].bar(labels, per_label_precision)
    axs[0, 0].set_title('Precision')
    axs[0, 0].set_xlabel('Labels')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_ylim([0, 1])
    
    # Plot recall
    axs[0, 1].bar(labels, per_label_recall)
    axs[0, 1].set_title('Recall')
    axs[0, 1].set_xlabel('Labels')
    axs[0, 1].set_ylabel('Recall')
    axs[0, 1].set_ylim([0, 1])
    
    # Plot F1-score
    axs[1, 0].bar(labels, per_label_f1)
    axs[1, 0].set_title('F1-score')
    axs[1, 0].set_xlabel('Labels')
    axs[1, 0].set_ylabel('F1-score')
    axs[1, 0].set_ylim([0, 1])
    
    # Plot accuracy
    per_label_accuracy = []
    for label in range(y_true.shape[1]):
        label_accuracy = accuracy_score(y_true[:, label], y_pred[:, label])
        per_label_accuracy.append(label_accuracy)
    
    axs[1, 1].bar(labels, per_label_accuracy)
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].set_xlabel('Labels')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_ylim([0, 1])
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Display the plots
    plt.show()

