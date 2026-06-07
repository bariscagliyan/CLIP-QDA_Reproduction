"""CLIP-QDA classifier.

This is a thin wrapper around scikit-learn's QuadraticDiscriminantAnalysis,
which is exactly the model described in Section 3.3 of the paper: a per-class
multivariate Gaussian N(z | mu_c, Sigma_c) with class priors p_c, classifying
by the maximum-a-posteriori rule (Eq. 4). ``store_covariance=True`` keeps the
per-class means/covariances so the XAI module can reuse them.
"""

import joblib
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score


class CLIPQDA:
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param
        self.model = QuadraticDiscriminantAnalysis(
            reg_param=reg_param, store_covariance=True)
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # statistical parameters used by the XAI methods 
    @property
    def means_(self):
        return np.asarray(self.model.means_)

    @property
    def priors_(self):
        return np.asarray(self.model.priors_)

    @property
    def covariances_(self):
        # sklearn stores a list of (n_features, n_features) arrays.
        return [np.asarray(c) for c in self.model.covariance_]

    def class_index(self, label):
        return int(np.where(self.classes_ == label)[0][0])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_eval(X_train, y_train, X_test, y_test, reg_param=1e-4):
    """Train a CLIP-QDA and return (model, metrics dict)."""
    clf = CLIPQDA(reg_param=reg_param).fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test) if len(X_test) else float("nan")
    metrics = {
        "reg_param": reg_param,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }
    return clf, metrics
