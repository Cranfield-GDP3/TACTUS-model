from typing import List, Union
from pathlib import Path
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from .torch_mlp import TorchMLP
from tactus_data import BodyAngles

AVAILABLE_MODELS = {
    "LSTM": "LSTM",
    "SVC": SVC,
    "TorchMLP": TorchMLP,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


class Classifier:
    def __init__(self, classifier_name: str = None, hyperparams: dict = None) -> None:
        """
        Create the classifier instance with the given name and
        hyperparametres.

        Parameters
        ----------
        classifier_name : str, optional
            name of the classifier. Must be in "SVC", "MLPClassifier",
            "GradientBoostingClassifier", by default None.
        hyperparams : dict, optional
            dictionnary of the classifier hyperparametres,
            by default None.
        """
        self.clf = None
        self.name = None
        self.hyperparams = None

        self.window_size: int = None
        self._angles_to_compute: List[BodyAngles] = None
        self.fps: int = None

        if classifier_name is not None and hyperparams is not None:
            self.name = classifier_name
            self.hyperparams = hyperparams
            self.clf = AVAILABLE_MODELS[classifier_name](**hyperparams)
        self.pca = None

    @classmethod
    def load(cls, model_weights_path: Path) -> "Classifier":
        """
        load model weights from a path.

        Parameters
        ----------
        model_weights_path : Path
            path to the model weights.

        Returns
        -------
        Classifier
            return instance of the classifier to allow chaining like
        Examples
        --------
        `clf = Classifier.load(model_weights_path)`.
        """
        return pickle.load(model_weights_path.open("rb"))

    def save(self, save_path: Path):
        """
        save the model weights inside a pickle file.

        Parameters
        ----------
        save_path : Path
            where to save the model weights.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self, save_path.open(mode="wb"))

    def fit_pca(self, X: Union[np.ndarray, List[List]], Y = None, *, min_pca_features: int = 50):
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Y :
            ignored
        min_pca_features : int, optional
            minimum number of features after the pca, by default 50.

        Returns
        -------
        PCA : object
            Returns the instance itself.
        """
        n_components = min(min_pca_features, X.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        return self.pca.fit(X)

    def transform(self, X: Union[np.ndarray, List[List]]):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where
            `n_samples` is the number of samples and `n_components` is
            the number of the components.
        """
        return self.pca.transform(X)

    def fit(self, X, Y):
        """
        fit the classifier to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            features.
        Y : array-like of shape (n_samples)
            ground truth.

        Returns
        -------
        self
            return the instance of the classifier.
        """
        return self.clf.fit(X, Y)

    def predict(self, X):
        """
        predict the label on a feature set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            features.

        Returns
        -------
        array-like of shape (n_samples)
            the predicted labels.
        """
        return self.clf.predict(X)

    @property
    def angles_to_compute(self) -> List[BodyAngles]:
        """for compatibility purpose."""
        if hasattr(self, "angles_to_compute"):
            return self.angles_to_compute
        else:
            DeprecationWarning("This instance has an depreciated attribute ",
                               "angle_to_compute. Use angles_to_compute instead.")
            return self.angle_to_compute

    @angles_to_compute.setter
    def angles_to_compute(self, values: List[BodyAngles]):
        self._angles_to_compute = values
