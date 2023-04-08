from typing import List, Union
from pathlib import Path
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA


AVAILABLE_MODELS = {
    "LSTM": "LSTM",
    "SVC": SVC,
    "MLPClassifier": MLPClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


class Classifier:
    def __init__(self, classifier: str = None, hyperparams: dict = None) -> None:
        """
        Create the classifier instance with the given name and
        hyperparametres.

        Parameters
        ----------
        classifier : str, optional
            name of the classifier. Must be in "SVC", "MLPClassifier",
            "GradientBoostingClassifier", by default None.
        hyperparams : dict, optional
            dictionnary of the classifier hyperparametres,
            by default None.
        """
        self.clf = None
        if classifier is not None and hyperparams is not None:
            self.clf = AVAILABLE_MODELS[classifier](**hyperparams)
        self.pca = None

    def load(self, model_weights_path: Path):
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
            `clf = Classifier().load(model_weights_path)`.
        """
        self.clf = pickle.load(model_weights_path.open("rb"))
        return self

    def save(self, save_path: Path):
        """
        save the model weights inside a pickle file.

        Parameters
        ----------
        save_path : Path
            where to save the model weights.
        """
        save_path.mkdir(parents=True, exist_ok=True)
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
