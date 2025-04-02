from annoy import AnnoyIndex
from sklearn.impute import KNNImputer
import numpy as np

class AnnoyKNNImputer:
    def __init__(self, n_neighbors=5, metric='euclidean', n_trees=10):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.annoy_indexer = None

    def fit(self, X, y=None):
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
                    nn_values = X[nn_indices, nan_index]
                    X_imputed[i, nan_index] = np.nanmean(nn_values)
        return X_imputed