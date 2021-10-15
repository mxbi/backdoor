# Main loop
import timm
import torch
import numpy as np
from tqdm import tqdm

import wandb
wandb.init(project='backdoor', entity='mxbi')

model = timm.create_model('resnet18', pretrained=True, num_classes=10)

import kmnist
from utils import totensor, tonp

dataset = kmnist.KuzushijiMNIST()
data = dataset.get_data()

print(f"Training set {data['train'][0].shape}, {data['train'][1].shape} Test set {data['test'][0].shape}, {data['test'][1].shape}")

class Trainer:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(), optimizer=torch.optim.SGD, optimizer_params={}, device='cuda'):
        self.model = model.to(device)
        self.device = device

        self.criterion = criterion
        self.optim = optimizer(self.model.parameters(), **optimizer_params)

        if 'lr' in optimizer_params:
            wandb.log({'lr': optimizer_params['lr']})

    def set_learning_rate(self, lr):
        for g in self.optim.param_groups:
            g['lr'] = lr
        wandb.log({'lr': lr})

    def get_mean_gradients(self):
        abs_gradient_sum = 0
        abs_gradient_count = 0
        for param in model.parameters():
            abs_grad = torch.abs(param.grad)
            abs_gradient_sum += torch.sum(abs_grad)
            abs_gradient_count += torch.prod(abs_grad)
        return abs_gradient_sum / abs_gradient_count

    def train_epoch(self, X, y, bs=64, shuffle=False):
        assert len(X) == len(y), "X and y must be the same length"
        self.model.train()
        n_batches = int(np.ceil(len(X) / bs))

        if shuffle:
            shuffle_ixs = np.random.permutation(np.arange(len(X)))
            X = X[shuffle_ixs]
            y = y[shuffle_ixs]

        for i_batch in tqdm(range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)

            self.optim.zero_grad()
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch.type(torch.cuda.LongTensor))

            # Accuracy measurement
            outputs_cpu = tonp(outputs)
            batch_acc = (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).mean()

            loss.backward()
            gradient_size = self.get_mean_gradients()

            wandb.log({"batch_loss": loss, "batch_lma_gradient": np.log(tonp(gradient_size)), "batch_acc": batch_acc})
            self.optim.step()

    def evaluate_epoch(self, X, y, bs=64):
        assert len(X) == len(y), "X and y must be the same length"
        self.model.eval()
        n_batches = int(np.ceil(len(X) / bs))

        total_loss = 0.
        total_acc = 0.
        for i_batch in tqdm(range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)
            
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch.type(torch.cuda.LongTensor))

            # Add loss on the batch to the accumulator
            total_loss += tonp(loss) * len(x_batch)

            # Measure accuracy on the batch and add to accumulator
            outputs_cpu = tonp(outputs)
            total_acc += (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).sum()

        # Get summary statistics over the whole epoch
        wandb.log({"eval_loss": total_loss / len(X), "eval_acc": total_acc / len(X)})

            
wandb.watch(model, log_freq=100)

t = Trainer(model, optimizer=torch.optim.Adam, optimizer_params={'lr': 0.001})
for i in range(100):
    t.set_learning_rate(0.001 * (0.95)**i)
    t.train_epoch(*data['train'], bs=128, shuffle=True)
    t.evaluate_epoch(*data['test'], bs=128)