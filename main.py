from Data_Pre_Processing.Data_loading import load_data, flatten_matrices
from Data_Pre_Processing.Data_splitting import train_val_test_split, create_tensordatasets, create_data_loaders
from model.model import SpectralNet, get_optimiser, criterion, scheduler
from train_eval.training import train, compute_gradient_norms
import torch
from train_eval.eval import evaluate

# Load the data
x_data = load_data("/Users/sanazkazeminia/Documents/Mass_Spec_project/Mass_Spec_ML_Project/spectra.pkl") # replace with actual path
y_data = load_data("/Users/sanazkazeminia/Documents/Mass_Spec_project/Mass_Spec_ML_Project/matrices.pkl") # replace with actual path
flattened_y = flatten_matrices(y_data)

# Split the data into train, validation, and test sets
x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x_data, flattened_y) # split 80/20 then 10/10

# Create TensorDatasets and DataLoaders
train_set, val_set, test_set, _ = create_tensordatasets(x_train, x_val, x_test, y_train, y_val, y_test) # noisy_test_set not used 
train_loader, val_loader,test_loader, _ = create_data_loaders(train_set, val_set, test_set, None, batch_size=256) # noisy_test_loader not used

# Initialize the model, optimizer, and other training components
model = SpectralNet()
lr = 0.000001
optimizer = get_optimiser(model, lr)
scheduler = scheduler(optimizer, base_lr=lr, max_lr=1e-4, step_size_up=2000, mode='triangular2')
criterion = criterion()
num_epochs = 1
threshold = 0.5

# Train the model
train_losses, val_losses, y_true_val, y_pred_val = train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, threshold)

y_true_test, y_pred_test, X_true_test, test_loss = evaluate(model, test_loader, criterion)
print("Evaluation on test set:", test_loss)


# Save the trained model
torch.save(model.state_dict(), "model_1.pth")
print("Trained model saved as model_1.pth")