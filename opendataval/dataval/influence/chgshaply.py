from functools import partial

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel
import copy
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier, DummyRegressor
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.autograd import grad

from opendataval.dataloader.util import CatDataset

class CHGShapley(DataEvaluator, ModelMixin):
    """CHG Data Shapley implementation.

    Parameters
    ----------
    grad_args : tuple, optional
        Positional arguments passed to the model.grad function
    grad_kwargs : dict[str, Any], optional
        Key word arguments passed to the model.grad function
    """

    def __init__(self,*grad_args, **grad_kwargs):
        self.args = grad_args
        self.kwargs = grad_kwargs

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Function Data Valuation.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.influence = np.zeros(len(x_train))
        return self

    def input_model(self, pred_model: GradientModel):
        """Input the prediction model with gradient.

        Parameters
        ----------
        pred_model : GradientModel
            Prediction model with a gradient
        """
        assert (  # In case model doesn't inherit but still wants the grad function
            isinstance(pred_model, GradientModel)
            or callable(getattr(pred_model, "grad"))
        ), ("Model with gradient required.")

        self.pred_model = pred_model.clone()
        return self
    def shap_value_evaluation(self, X, alpha, loss): 
        with torch.no_grad():
            n, d = X.shape
            x_sum = torch.sum(X, dim=0)
            
            sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
            sum_2 = torch.sum(1 / torch.arange(1, n + 1).float()**2)
            term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
            term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
            term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
            term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
            term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
            term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)
            # term_7 = 1 / (n - 1) * (sum_1 - 1 / n) * loss - 1 / (n * (n - 1)) * (sum_1 - 1) * torch.sum(loss)

            shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6)
            # shapley_values -= shapley_values.min()
            # shapley_values /= shapley_values.sum()
            # term_7 -= term_7.min()
            # term_7 /= term_7.sum()
            # term_7 = term_7.max() - term_7
            # shapley_values *= term_7
            # shapley_values *= loss
            # shapley_values -= shapley_values.min()
            # shapley_values /= shapley_values.sum()
            return shapley_values
    
    def train_data_values(self, *args, **kwargs):
        """Trains model to compute Data Shapley of each data point (data values).

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Trains model on training data so we can find gradients of trained model
        batch_size = kwargs["batch_size"]
        epochs = kwargs["epochs"]
        lr = kwargs["lr"]
        sample_weight = None
        optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=lr)
        criterion = F.binary_cross_entropy if self.pred_model.num_classes == 2 else F.cross_entropy
        
        dataset = CatDataset(self.x_train, self.y_train, sample_weight)
        valid_dataset = CatDataset(self.x_valid, self.y_valid, sample_weight)
        self.pred_model.train()
        for _ in range(int(epochs)):
            train_grad_list = []
            valid_grad_list = []
            train_loss_list = []
            valid_loss_list = []
            for x_batch, y_batch, *weights in DataLoader(
                dataset, batch_size=1, shuffle=False, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.pred_model.device)
                y_batch = y_batch.to(device=self.pred_model.device)
                # outputs = self.pred_model.predict(x_batch)
                outputs = self.pred_model.__call__(x_batch)
                batch_loss = criterion(outputs, y_batch, reduction="mean")
                # batch_grad = torch.autograd.grad(loss, outputs)[1] ##only bias
                batch_grad = grad(batch_loss, self.pred_model.parameters())[1]
                train_grad_list.append(batch_grad)
                train_loss_list.append(batch_loss)

            for x_batch, y_batch, *weights in DataLoader(
                valid_dataset, batch_size=1, shuffle=False, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.pred_model.device)
                y_batch = y_batch.to(device=self.pred_model.device)
                outputs = self.pred_model.__call__(x_batch)

                batch_loss = criterion(outputs, y_batch, reduction="mean")
                batch_grad = grad(batch_loss, self.pred_model.parameters())[1]##only bias
                valid_grad_list.append(batch_grad)
                valid_loss_list.append(batch_loss)
            valid_grad_tensor = torch.stack(valid_grad_list, dim=0)
            train_grad_tensor = torch.stack(train_grad_list, dim=0)
            valid_loss_tensor = torch.tensor(valid_loss_list)
            train_loss_tensor = torch.tensor(train_loss_list)
            
            train_grad_tensor = train_loss_tensor[:, None] * train_grad_tensor
            valid_grad_tensor = valid_loss_tensor[:, None] * valid_grad_tensor

            # train_grad_tensor = torch.cat([train_grad_tensor, valid_grad_tensor], dim=0)
            # Calculate mean gradient for validation data
            # valid_grad_mean = torch.mean(valid_grad_tensor, dim=0)
            train_grad_mean = torch.mean(train_grad_tensor, dim=0)
            # valid_grad_mean = (len(train_loss_list) * train_grad_mean
            #                    +len(valid_loss_list) * valid_grad_mean)/(len(train_loss_list)+len(valid_loss_list))
            valid_grad_mean = train_grad_mean                   
            self.influence += np.array(self.shap_value_evaluation(train_grad_tensor, valid_grad_mean, train_loss_tensor)[0:len(train_loss_list)].cpu())

            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.pred_model.device)
                y_batch = y_batch.to(device=self.pred_model.device)

                optimizer.zero_grad()
                outputs = self.pred_model.__call__(x_batch)

                if sample_weight is not None:
                    # F.cross_entropy doesn't support sample_weights
                    loss = criterion(outputs, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.pred_model.device)).mean()
                else:
                    loss = criterion(outputs, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

        self.influence /= epochs
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return influence (data values) for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values for training input data point
        """
        return self.influence
