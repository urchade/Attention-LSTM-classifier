import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm


class ClassifierTrainer:
    """Python class for training the lstm text classifier"""

    def __init__(self, model, train_loader, val_loader, loss_function=nn.NLLLoss(),
                 metric=accuracy_score, optimizer=torch.optim.Adam, **optim_kwargs):
        """
        Parameters
        ----------
        model: nn.Module
            the model to train
        train_loader: pytorch Dataloader
        val_loader: pytorch Dataloader
        loss_function: nn.Module
        optimizer: pytorch optimizer
        metric: callable
            A metric; ex: sklearn.metrics.accuracy_score
        """

        # Storing some stuffs ...
        self.model = model
        self.loss_fun = loss_function
        self.optimizer = optimizer(self.model.parameters(), **optim_kwargs)
        self.metric = metric

        self.device = next(self.model.parameters()).device
        self.train_data = train_loader
        self.val_data = val_loader

        self.val_losses = []
        self.train_losses = []

        # best model state dict obtain with the early stopping algorithm
        self.best_state_dict = None
        self.optimal_num_epochs = None

        self.train_metric = []
        self.val_metric = []

    @torch.no_grad()
    def evaluate(self, data=None):
        """
        A function for evaluating the model
        Parameters
        ----------
        data
        Returns
        -------
        Mean loss/Metric value for an epoch
        Tuple (y_true, y_pred)
        """
        self.model.eval()

        val_errors = []
        pred_ids = []
        true_ids = []

        if data is None:
            data = self.val_data

        for x, y, mask in data:
            y_hat = self.model.forward(x.to(self.device), mask.to(self.device))
            loss = self.loss_fun(y_hat, y.to(self.device))
            val_errors.append(loss.item())

            pred_ids.append(self.model.predict(y_hat, logits=True))
            true_ids.append(y.numpy())

        pred_ids = [a for i in pred_ids for a in i]
        true_ids = [a for i in true_ids for a in i]

        return np.mean(val_errors), self.metric(true_ids, pred_ids), (true_ids, pred_ids)

    def update(self, data=None):
        """
        A function for updating the model
        Parameters
        ----------
        data
        Returns
        -------
        Mean loss/Metric value for an epoch
        """

        self.model.train()

        train_errors = []
        pred_ids = []
        true_ids = []

        if data is None:
            data = self.train_data

        for x, y, mask in data:
            y_hat = self.model.forward(x.to(self.device), mask.to(self.device))
            loss = self.loss_fun(y_hat, y.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_ids.append(self.model.predict(y_hat, logits=True))
            true_ids.append(y.numpy())
            train_errors.append(loss.item())

        pred_ids = [id for list_ids in pred_ids for id in list_ids]
        true_ids = [id for list_ids in true_ids for id in list_ids]

        return np.mean(train_errors), self.metric(true_ids, pred_ids)

    def train(self, max_epoch=10, patience=3):
        """
        Parameters
        ----------
        max_epoch: Maximum number of epochs
        patience: Patience for early stopping
        """

        best_metric = 0
        j = 0

        for i in tqdm(range(max_epoch)):

            train_loss, train_metric = self.update()
            val_loss, val_metric, _ = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metric.append(train_metric)
            self.val_metric.append(val_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                self.best_state_dict = self.model.state_dict()
                self.optimal_num_epochs = i
                j = 0
            else:
                j += 1

            if j > patience:
                print("Early stopping ...")
                break

    def plot_loss_curve(self):
        """Plotting the loss curve"""
        sns.set_style("darkgrid")
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title("Loss curve")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.show()

    def plot_metric_curve(self):
        """Plotting the metric curve"""
        sns.set_style("darkgrid")
        plt.plot(self.train_metric, label='Train')
        plt.plot(self.val_metric, label='Validation')
        plt.axvline(x=self.optimal_num_epochs, ls='--')
        plt.title("Metric curve")
        plt.xlabel('Epochs')
        plt.ylabel('Metric values')
        plt.legend()
        plt.show()

    def report_result(self, data_loader=None):
        """
        A method for printing a classification report given a data loader
        """
        if data_loader is None:
            data_loader = self.val_data

        _, _, (y_true, y_pred) = self.evaluate(data_loader)

        report = classification_report(y_true, y_pred)

        print(report)

    def plot_cm(self, data_loader=None):
        """
        A method for plotting confusion matrix given a data loader
        """
        if data_loader is None:
            data_loader = self.val_data
        _, _, (y_true, y_pred) = self.evaluate(data_loader)
        cm = metrics.confusion_matrix(y_true, y_pred)
        idx = sorted(np.unique(y_true))
        confusion_matrix = pd.DataFrame(cm, index=idx, columns=idx)
        plt.figure(figsize=(7, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt=".0f", square=True, cbar=False)
        plt.show()
