# Main loop
import timm
import torch
import numpy as np

model = timm.create_model('resnet18', pretrained=True)

import kmnist

dataset = kmnist.KuzushijiMNIST()
data = dataset.get_data()

print(f"Training set {data['train'][0].shape}, {data['train'][1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")

class Trainer:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(), optimizer=torch.optim.SGD, optimizer_params={}):
        self.model = model

        self.criterion = criterion
        self.optim = optimizer(self.model.parameters(), **optimizer_params)

    def train_epoch(self, X, y, bs=64):
        assert len(X) == len(y), "X and y must be the same length"
        n_batches = len(X) // bs + 1
        for i in range(n_batches + 1):
            
