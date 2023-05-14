# Code from https://github.com/JonathanCrabbe/CARs

import torch
from torch import nn
from torchvision.models import inception_v3
from typing import Optional
from tqdm import tqdm
import numpy as np
import pathlib
import os

class CUBInception(nn.Module):
    def __init__(self, name: str = "inception_model"):
        super().__init__()
        self.inception = inception_v3(pretrained=True)
        self.fc = nn.Linear(2048, 200)
        self.criterion = nn.CrossEntropyLoss()
        self.name = name

    def forward(self, x):
        x = self.input_to_representation(x)
        x = self.fc(x)
        # N x 200
        return x

    def input_to_representation(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.inception.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

    def mixed_7b_rep(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.inception.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        return x

    def representation_to_output(self, h):
        return self.fc(h)

    def mixed_7b_rep_to_output(self, h):
        # N x 2048 x 8 x 8
        h = self.inception.Mixed_7c(h)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        h = self.inception.avgpool(h)
        # N x 2048 x 1 x 1
        h = self.inception.dropout(h)
        # N x 2048 x 1 x 1
        h = torch.flatten(h, 1)
        # N x 2048
        return self.fc(h)

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer

        Returns:
            average loss on the training set
        """
        self.train()
        loss_meter = AverageMeter("Loss")
        train_loss = []
        for image_batch, label_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(image_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), image_batch.size(0))
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        One epoch of the testing loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for image_batch, label_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(image_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1)).cpu().numpy() / len(label_batch)
                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            lr: int = 1e-03, n_epoch: int = 50, patience: int = 50, checkpoint_interval: int = -1) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save

        Returns:

        """
        self.to(device)
        #optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)
        optim = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=4e-05, momentum=.9)
        waiting_epoch = 0
        best_test_acc = 0
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            print(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train Loss {train_loss:.3g} \t '
                         f'Test Loss {test_loss:.3g} \t'
                         f'Test Accuracy {test_acc * 100:.3g}% \t ')
            if test_acc <= best_test_acc:
                waiting_epoch += 1
                print(f'No improvement over the best epoch \t Patience {waiting_epoch} / {patience}')
            else:
                print(f'Saving the model in {save_dir}')
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_acc = test_acc.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                print(f'Saving checkpoint {n_checkpoint} in {save_dir}')
                path_to_checkpoint = save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                print(f'Early stopping activated')
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        path_to_model = os.path.join(directory,self.name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def get_hooked_modules(self) -> dict:
        return {
            "Mixed5d": self.inception.Mixed_5d, "Mixed6e": self.inception.Mixed_6e,
            "Mixed7c": self.inception.Mixed_7c, "Mixed7b": self.inception.Mixed_7b,
            "InceptionOut": self.inception.avgpool
               }
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count