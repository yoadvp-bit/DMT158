from annoy import AnnoyIndex
import numpy as np
import pickle


class AnnoyKNNImputer:
    def __init__(self, n_neighbors=5, metric='euclidean', n_trees=10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.annoy_indexer = None
        self.X_train = None  # <== store reference data

    def fit(self, X, y=None):
        self.X_train = X  # Store full training data
        self.annoy_indexer = AnnoyIndex(X.shape[1], self.metric)
        for i, row in enumerate(X):
            self.annoy_indexer.add_item(i, row.tolist())
        self.annoy_indexer.build(self.n_trees)
        return self

    def transform(self, X):
        X_imputed = X.copy()
        for i, row in enumerate(X):
            nan_indices = np.where(np.isnan(row))[0]
            if len(nan_indices) > 0:
                nn_indices = self.annoy_indexer.get_nns_by_vector(row.tolist(), self.n_neighbors)
                for nan_index in nan_indices:
                    nn_values = self.X_train[nn_indices, nan_index]  # Fix: reference global X_train
                    X_imputed[i, nan_index] = np.nanmean(nn_values)
        return X_imputed
    
    def save(self, path, dim):
        if self.annoy_indexer is not None:
            self.annoy_indexer.save(f"{path}.ann")
            with open(f"{path}_meta.pkl", "wb") as f:
                pickle.dump({
                    "n_neighbors": self.n_neighbors,
                    "metric": self.metric,
                    "n_trees": self.n_trees,
                    "dim": dim
                }, f)

    def load(self, path):
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.n_neighbors = meta["n_neighbors"]
            self.metric = meta["metric"]
            self.n_trees = meta["n_trees"]
            dim = meta["dim"]
            self.annoy_indexer = AnnoyIndex(dim, self.metric)
            self.annoy_indexer.load(f"{path}.ann")