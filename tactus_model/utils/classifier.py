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
    def __init__(self, classifier: str, hyperparams: dict) -> None:
        self.clf = AVAILABLE_MODELS[classifier](**hyperparams)
        self.pca = None

    def load(self, model_weights_path: Path):
        self.clf = pickle.load(model_weights_path.open("rb"))

    def save(self, save_path: Path):
        pickle.dump(self, save_path.open(mode="rb"))

    def fit_pca(self, X: np.ndarray, min_pca_features: int = 50):
        n_components = min(min_pca_features, X.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(X)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        self.clf.predict(X)

    def predict_proba(self, X):
        self.clf.predict_proba(X)
