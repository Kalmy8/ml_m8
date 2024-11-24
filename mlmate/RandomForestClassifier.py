from numpy.random import default_rng
import numpy as np
import pandas as pd

from mlmate.DecisionTreeClassifier import DTC

class RForest_Classifier:
    def __init__(self, n_trees, restrictions: dict = None):
        self.n_trees = n_trees
        self.restrictions = restrictions
        self.trees = [None] * n_trees

    def fit(self, dataset: pd.DataFrame, target_name: str):
        for i in range(self.n_trees):
            new_tree = DTC(self.restrictions)
            new_tree.fit(self._boostrap_dataset(dataset, target_name), target_name)
            self.trees[i] = new_tree

    def predict(self, X: pd.DataFrame):
        all_predictions = pd.Series(index=X.index)

        for tree in self.trees:
            features_seen = tree.features_seen
            prediction = tree.predict(X)
            all_predictions = pd.concat([all_predictions, prediction], axis=1)

        return all_predictions.mode(axis=1)

    @staticmethod
    def _boostrap_dataset(dataset, target_name: str):
        X = dataset.drop(target_name, axis=1)
        n_samples = X.shape[0]
        n_features = int(np.ceil(np.sqrt(X.shape[1])))

        rng = default_rng()
        random_features = rng.choice(X.shape[1], size=n_features, replace=False)

        random_samples = np.random.randint(n_samples, size=n_samples)

        return X.iloc[random_samples, random_features].join(dataset[target_name])