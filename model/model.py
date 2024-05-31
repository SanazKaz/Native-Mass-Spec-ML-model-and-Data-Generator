import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from timm.loss import AsymmetricLossMultiLabel
import logging
import os
import torch


logging.basicConfig(level=logging.INFO)

class SpectralNet(nn.Module):
    def __init__(self):
        super(SpectralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2001, 1024), # 2001 is the number of features in the spectra data (change with binning)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256), ### add dropout at some point
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Added layer
            nn.ReLU(),  # Added activation function
            nn.Linear(128, 36), # 36 is the number of features in the interaction matrix (change with size of matrix)
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def get_model():
    model = SpectralNet()
    return model

def get_optimiser(model, lr):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    return optimiser

def scheduler(optimiser, base_lr, max_lr, step_size_up, mode):
    scheduler = CyclicLR(optimiser, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode=mode)
    return scheduler

def criterion():
    criterion = nn.BCELoss() # binary cross entropy loss for multilabel classification
    return criterion

model = SpectralNet()

def assym_criterion(gamma_neg, gamma_pos, clip):
    """for highly imbalanced set where more weight is required for the positive/ neg class"""
    criterion = AsymmetricLossMultiLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
    return assym_criterion


def load_checkpoint(model, optimizer, loss, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  
    # This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss

logging.info("model script executed successfully")


if __name__ == "__main__":
    model = SpectralNet()
    dummy_input = torch.randn(1, 2001)  # Create a dummy input tensor
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")