import torch
import numpy as np
from tqdm import tqdm

from .utils import totensor, tonp

# TODO: Adjustable logging
import wandb

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
        for param in self.model.parameters():
            abs_grad = torch.abs(param.grad)
            abs_gradient_sum += torch.sum(abs_grad)
            abs_gradient_count += torch.prod(abs_grad)
        return abs_gradient_sum / abs_gradient_count

    def train_epoch(self, X, y, bs=64, shuffle=False, name='train'):
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

            wandb.log({f"{name}_batch_loss": loss, f"{name}_batch_lma_gradient": np.log(tonp(gradient_size)), f"{name}_batch_acc": batch_acc})
            self.optim.step()

    def evaluate_epoch(self, X, y, bs=64, name='eval'):
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
        epoch_metrics = {f"{name}_eval_loss": total_loss / len(X), f"{name}_eval_acc": total_acc / len(X)}
        wandb.log(epoch_metrics)

        return epoch_metrics