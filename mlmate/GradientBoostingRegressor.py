from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes


class DummyRegressor:
    def __init__(self):
        self.learned_mean = None

    def fit(self, y: np.ndarray):
        self.learned_mean = np.mean(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = np.empty((X.shape[0]), dtype=float)
        prediction[:] = self.learned_mean

        return prediction


def dMSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 0.5 * (y_true - y_pred)


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return sum((y_true - y_pred)**2)


class GradientBoostingRegressor:
    '''
    MSE loss and MSE criterion
    '''

    def __init__(self,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1
                 ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.base_estimator = DummyRegressor()
        self.ensemble: list[Any] = [self.base_estimator]

        self.tree = RandomForestRegressor

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Fit base estimator
        self.ensemble[0].fit(y)

        while len(self.ensemble) < self.n_estimators:
            prediction = self.predict(X)
            print(f"MSE loss is now {MSE(y, prediction)}")
            grad = dMSE(y, prediction)
            new_base_estimator = self.tree(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf).fit(X, grad)
            self.ensemble.append(new_base_estimator)

    def predict(self, X: np.ndarray) -> np.ndarray:
        ensemble_predictions = np.hstack([base_est.predict(X).reshape(-1,1) for base_est in self.ensemble])
        learning_rate_vector = np.empty((ensemble_predictions.shape[1], 1), dtype=float)
        learning_rate_vector[:, :] = self.learning_rate
        learning_rate_vector[0,0] = 1

        return ensemble_predictions @ learning_rate_vector


if __name__ == '__main__':
    # Load the diabetes dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

    gbr = GradientBoostingRegressor()
    gbr.fit(diabetes_X, diabetes_y)

    y_pred = gbr.predict(diabetes_X)
    print(f"MSE error is {MSE(y_true = diabetes_y, y_pred = y_pred)}")