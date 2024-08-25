"""Trainer class for AlexNet"""

import sys, os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "alexnet",
        batch_size: int = 256,
        learning_rate: float = 0.01,
        num_epochs: int = 30,
        check_val_every_n_epoch: int = 1,
        device: str = "cpu",
        checkpoints_dir: str = "checkpoints",
    ) -> None:
        """Trainer object to facilitate training and evaluation"""

        self.model = model
        self.model_name = model_name

        # training configurations
        self.batch_size = batch_size  # does nothing; mainly for viz
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.device = device
        self.model.to(self.device)

        # set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()

        # SGD used by original paper "Deep Residual Learning for Image Recognition"
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        # model metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

        # create checkpoints directory
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)  # create plots dir

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """Train the AlexNet Model"""

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
                running_acc += self.__accuracy(outputs, labels)

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

            self.scheduler.step()

            train_accuracy = running_acc / len(train_dataloader)
            self.train_accuracies.append((epoch, train_accuracy.cpu()))

            avg_loss = running_loss / len(train_dataloader)
            self.train_losses.append((epoch, avg_loss))

            if epoch % self.check_val_every_n_epoch == 0:
                self.model.eval()  # set model to evaluation
                with torch.no_grad():
                    running_val_acc = 0
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        running_vloss += loss.item()
                        # compute validtion accuracy
                        running_val_acc += self.__accuracy(outputs, labels)

                val_accuracy = running_val_acc / len(val_dataloader)
                self.val_accuracies.append((epoch, val_accuracy.cpu()))

                avg_vloss = running_vloss / len(val_dataloader)
                self.val_losses.append((epoch, avg_vloss))

                self.logger.info(
                    f"[EPOCH {epoch + 1}] LOSS : train={avg_loss} val={avg_vloss} | ACCURACY : train={train_accuracy} val={val_accuracy}"
                )

    def test(self, test_dataloader: DataLoader) -> None:
        """Test the AlexNet Model"""

        correct = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                correct += self.__accuracy(outputs, labels)

        self.logger.info(f"Test accuracy: {(correct / len(test_dataloader)) * 100} %")

    def plot_metrics(self) -> None:
        """Create plots for model metrics"""

        os.makedirs("plots", exist_ok=True)  # create plots dir

        t_iters, t_loss = list(zip(*self.train_losses))
        _, v_loss = list(zip(*self.val_losses))
        _, acc = list(zip(*self.train_accuracies))
        _, v_acc = list(zip(*self.val_accuracies))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Model: [{self.model_name}]")

        ax[0].set_title(f"Loss Curve (batch_size={self.batch_size}, lr={self.learning_rate})")
        ax[0].plot(t_iters, t_loss)
        ax[0].plot(t_iters, v_loss)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(["Train", "Validation"])
        ax[0].set_xticks(t_iters)

        ax[1].set_title(f"Accuracy Curve (batch_size={self.batch_size}, lr={self.learning_rate})")
        ax[1].plot(t_iters, acc)
        ax[1].plot(t_iters, v_acc)
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(["Train", "Validation"])
        ax[1].set_xticks(t_iters)

        fig.savefig(f"plots/{self.model_name}_metrics.png")
        plt.show()

    def __accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy given outputs as logits"""

        preds = torch.argmax(outputs, dim=1)
        return torch.sum(preds == labels) / len(preds)

    def save(self, epoch: int, loss: float) -> None:
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.model_name}_e{epoch}_{time}.pt")
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_name: str) -> None:
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
