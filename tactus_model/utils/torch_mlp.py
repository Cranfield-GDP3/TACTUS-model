from typing import Tuple, List
import torch
import torch.nn as nn
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import _num_features


class TorchMLP:
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int],
        activation: str,
        *,
        dropout_layers: Tuple[int] = None
    ):
        if dropout_layers is None:
            dropout_layers = [0] * len(hidden_layer_sizes)
        if len(hidden_layer_sizes) != len(dropout_layers):
            raise IndexError('dropout_layers must be the same size as hidden_layer_sizes')

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_layers = dropout_layers

        self.activation = activation

        self.model = None
        self.loss_fn = None
        self.optimizer = None

    def fit(self, X, Y, *, num_epochs: int):
        """
        fit the model

        Parameters
        ----------
        num_epochs : int
            number of epochs
        """
        input_size = _num_features(X)
        output_size = len(unique_labels(Y))
        self._build(input_size, output_size)

        if isinstance(X, List):
            X = torch.Tensor(X)
        if isinstance(Y, List):
            Y = torch.LongTensor(Y)

        for _ in range(num_epochs):
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self. optimizer.step()

    def predict(self, X):
        """
        predict using the fitted model
        """
        if isinstance(X, List):
            X = torch.Tensor(X)

        return self.model(X)

    def _build(self, input_size: int, output_size: int):
        """
        automatically build the model in regard of the input and the
        output size.

        Parameters
        ----------
        input_size : int
            size of a feature
        output_size : int
            number of different classes
        """
        layer_sizes = [input_size]
        layer_sizes.extend(self.hidden_layer_sizes)
        layer_sizes.append(output_size)

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                activation_layer = getattr(nn, self.activation)
                layers.append(activation_layer())
                layers.append(nn.Dropout(self.dropout_layers[i]))

        layers.append(nn.Softmax())
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
