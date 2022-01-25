import torch
import numpy as np
from tqdm import tqdm

from .utils import totensor, tonp
from .image_utils import ImageFormat

# TODO: Adjustable logging
import wandb

class Trainer:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), 
                optimizer=torch.optim.SGD, optimizer_params={'lr': 0.01}, device='cuda', use_wandb=True,
                convert_image_format=True):
        self.model = model.to(device)
        self.device = device

        self.criterion = criterion
        self.optim = optimizer(self.model.parameters(), **optimizer_params)
        self.convert_image_format = convert_image_format

        # whether to enable wandb logging
        self.wandb = use_wandb

        if 'lr' in optimizer_params and self.wandb:
            wandb.log({'lr': optimizer_params['lr']})

    def set_learning_rate(self, lr):
        for g in self.optim.param_groups:
            g['lr'] = lr
        if self.wandb:
            wandb.log({'lr': lr})

    def get_mean_gradients(self):
        abs_gradient_sum = 0
        abs_gradient_count = 0
        for param in self.model.parameters():
            abs_grad = torch.abs(param.grad)
            abs_gradient_sum += torch.sum(abs_grad)
            abs_gradient_count += torch.prod(abs_grad)
        return abs_gradient_sum / abs_gradient_count

    def batch_inference(self, X):
        self.model.eval()

        if self.convert_image_format:
            X = ImageFormat.torch(X)

        return self.model(totensor(X, device=self.device))

    def train_epoch(self, X, y, sample_weights=None, bs=64, shuffle=False, name='train', progress_bar=True, tfm=None):
        assert len(X) == len(y), "X and y must be the same length"
        self.model.train()
        n_batches = int(np.ceil(len(X) / bs))

        # Convert image format to conform to training process (if necessary)
        if self.convert_image_format:
            X = ImageFormat.torch(X)

        # Randomly shuffle if required
        if shuffle:
            shuffle_ixs = np.random.permutation(np.arange(len(X)))
            X = X[shuffle_ixs]
            y = y[shuffle_ixs]

            if sample_weights is not None:
                sample_weights = sample_weights[shuffle_ixs]

        # Main loop
        for i_batch in (tqdm(range(n_batches)) if progress_bar else range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)

            if tfm:
                x_batch = tfm(x_batch)

            # print(x_batch.min(), x_batch.max())

            self.optim.zero_grad()
            outputs = self.model(x_batch)

            if sample_weights is not None:
                if self.criterion.reduction != 'none':
                    raise ValueError("Trying to use `sample_weights` with a reduced criterion. Use reduction='none' when specifying the criterion to allow sample weights to be applied.")

                loss = self.criterion(outputs, y_batch.type(torch.cuda.LongTensor))

                w_batch = totensor(sample_weights[i_batch*bs:(i_batch+1)*bs], device=self.device)
                loss = (loss * w_batch).mean()
            else:
                loss = self.criterion(outputs, y_batch.type(torch.cuda.LongTensor))
                # print(loss)
                # If the user specifies criterion with reduction=none, this means it can be used both with and without sample weights.
                if self.criterion.reduction == 'none':
                    loss = loss.mean()

            # Accuracy measurement
            outputs_cpu = tonp(outputs)
            batch_acc = (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).mean()

            loss.backward()
            gradient_size = self.get_mean_gradients()
            self.optim.step()
            
            if self.wandb:
                wandb.log({f"{name}_batch_loss": loss, f"{name}_batch_lma_gradient": np.log(tonp(gradient_size)), f"{name}_batch_acc": batch_acc})

    def evaluate_epoch(self, X, y, bs=64, name='eval', progress_bar=True):
        assert len(X) == len(y), "X and y must be the same length"
        self.model.eval()
        n_batches = int(np.ceil(len(X) / bs))

        if self.convert_image_format:
            X = ImageFormat.torch(X)

        total_loss = 0.
        total_acc = 0.
        for i_batch in (tqdm(range(n_batches)) if progress_bar else range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)

            outputs = self.model(x_batch)

            loss = self.criterion(outputs, y_batch.type(torch.cuda.LongTensor))
            # If the user specifies criterion with reduction=none, allow eval to still work.
            if self.criterion.reduction == 'none':
                    loss = loss.mean()

            # Add loss on the batch to the accumulator
            total_loss += tonp(loss) * len(x_batch)

            # Measure accuracy on the batch and add to accumulator
            outputs_cpu = tonp(outputs)
            total_acc += (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).sum()

        # Get summary statistics over the whole epoch
        epoch_metrics = {f"{name}_loss": total_loss / len(X), f"{name}_acc": total_acc / len(X)}
        if self.wandb:
            wandb.log(epoch_metrics)

        return epoch_metrics