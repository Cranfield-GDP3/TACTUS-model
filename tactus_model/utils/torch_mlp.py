from typing import Tuple, List
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import _num_features


class TorchMLP:
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int],
        activation: str,
        *,
        dropout_layers: Tuple[int] = None,
        num_epochs: Tuple[int] = None,
        batch_size: Tuple[int] = None,
        device: str = None,
    ):
        if dropout_layers is None:
            dropout_layers = [0] * len(hidden_layer_sizes)
        if len(hidden_layer_sizes) != len(dropout_layers):
            raise IndexError('dropout_layers must be the same size as hidden_layer_sizes')

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_layers = dropout_layers

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.model = None
        self.loss_fn = None
        self.optimizer = None

        self.select_device(device)

    def fit(self, X, Y, *, num_epochs: int = None, batch_size: int = None):
        """
        fit the model

        Parameters
        ----------
        num_epochs : int
            number of epochs
        """
        num_epochs = num_epochs or self.num_epochs
        batch_size = batch_size or self.batch_size
        if num_epochs is None:
            raise ValueError("num_epochs should be either defined as an instance attribute "
                             "or in the the fit function")
        if batch_size is None:
            raise ValueError("batch_size should be either defined as an instance attribute "
                             "or in the the fit function")

        input_size = _num_features(X)
        output_size = len(unique_labels(Y))

        self._build(input_size, output_size)

        if isinstance(X, List):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(Y, List):
            Y = torch.tensor(Y, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in tqdm(range(num_epochs)):
            for _, (x_batch, y_batch) in enumerate(dataloader):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self. optimizer.step()

    def predict(self, X):
        """
        predict using the fitted model
        """
        if isinstance(X, List):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

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

        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers).to(device=self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def select_device(self, device: str = None):
        """
        select the device or default to CPU if not found.

        Parameters
        ----------
        device : str, optional
            name of the device to use, by default None
        """
        if device is None:
            device = "cuda:0"

        device = device.lower()
        if not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)
