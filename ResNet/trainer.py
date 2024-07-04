import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        check_val_every_n_epoch: int = 1,
        device: str = "cpu",
    ) -> None:
        self.model = model

        # training configurations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.device = device
        self.model.to(self.device)

        # set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # model metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # logging info
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:

        for epoch in range(self.num_epochs):
            self.model.train()  # set model to train

            # loss tracking metrics
            running_loss = 0.0
            running_vloss = 0.0
            batch_loss = 0.0
            running_acc = 0.0

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero gradients for every batch
                self.optimizer.zero_grad()

                # compute predictions + loss
                outputs = self.model(inputs)  # predicted class
                loss = self.criterion(outputs, labels)

                # compute training accuracy
                running_acc += torch.sum(torch.all(outputs == labels, dim=1)) / labels.shape[0]

                # perform backpropagation
                loss.backward()  # compute gradients
                self.optimizer.step()  # update model parameters

                # gather data and report
                running_loss += loss.item()
                batch_loss += loss.item()
                if i % 10 == 0:
                    batch_loss = batch_loss / 10  # loss per batch
                    pbar.set_postfix({"loss": round(batch_loss, 5)})
                    batch_loss = 0.0

            train_accuracy = running_acc / len(train_dataloader)
            self.train_accuracies.append((epoch, train_accuracy.cpu()))

            avg_loss = running_loss / len(train_dataloader)
            self.train_losses.append((epoch, avg_loss))

            if epoch % self.check_val_every_n_epoch == 0:
                print("Validation...")
                self.model.eval()  # set model to evaluation
                with torch.no_grad():
                    running_val_acc = 0
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        # compute validtion accuracy
                        running_val_acc += torch.sum(torch.all(outputs == labels, dim=1)) / labels.shape[0]

                val_accuracy = running_val_acc / len(val_dataloader)
                self.val_accuracies.append((epoch, val_accuracy.cpu()))

                avg_vloss = running_vloss / len(val_dataloader)
                self.val_losses.append((epoch, avg_vloss))

                print(
                    f"[EPOCH {epoch}] LOSS : train={avg_loss} val={avg_vloss} | ACCURACY : train={train_accuracy} val={val_accuracy}"
                )

    def test(self, test_dataloader: DataLoader) -> None:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10000 test images: {100 * correct / total} %")
