"""
Utility functions for tensor decision trees.
"""

import numpy as np


# Distance functions
L2_norm = lambda A, B: np.linalg.norm(A - B, ord=2)


class custom_kmeans:
    """
    K-means clustering for tensor data.
    
    Custom implementation that works with arbitrary tensor shapes
    using a user-defined distance function.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    dist_func : callable, default=L2_norm
        Distance function between tensors
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    """
    
    def __init__(self, n_clusters, dist_func=L2_norm, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
    
    def fit(self, X):
        """
        Fit k-means clustering.
        
        Parameters
        ----------
        X : list of ndarrays
            List of tensor samples
        """
        if len(X) < self.n_clusters:
            raise ValueError("Number of samples must be >= n_clusters")
        
        # Randomly initialize centroids
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        centroids = [X[i] for i in indices]
        
        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            labels = self._assign_labels(X, centroids)
            
            # Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_samples = [x for x, label in zip(X, labels) if label == i]
                if len(cluster_samples) > 0:
                    new_centroids.append(np.mean(cluster_samples, axis=0))
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(centroids[i])
            
            # Check convergence
            converged = all(
                self.dist_func(old, new) < self.tol 
                for old, new in zip(centroids, new_centroids)
            )
            
            centroids = new_centroids
            
            if converged:
                break
        
        self.centroids = centroids
    
    def predict(self, X):
        """
        Predict cluster labels.
        
        Parameters
        ----------
        X : list of ndarrays
            List of tensor samples
            
        Returns
        -------
        labels : ndarray
            Cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        if len(X) == 0:
            return np.array([])
        
        return self._assign_labels(X, self.centroids)
    
    def _assign_labels(self, X, centroids):
        """Assign each sample to nearest centroid."""
        distances = np.array([
            [self.dist_func(x, c) for c in centroids]
            for x in X
        ])
        return distances.argmin(axis=1)


def compute_tensor_variance(X):
    """
    Compute variance along each spatial dimension of tensor.
    
    Parameters
    ----------
    X : ndarray
        Tensor data of shape (n_samples, *spatial_dims)
        
    Returns
    -------
    variances : ndarray
        Variance at each spatial location
    """
    return np.var(X, axis=0)


def leverage_score_sampling(X, sample_rate=0.01):
    """
    Sample spatial locations based on leverage scores (variance).
    
    Parameters
    ----------
    X : ndarray
        Tensor data
    sample_rate : float
        Proportion of locations to sample
        
    Returns
    -------
    sampled_indices : list of tuples
        Sampled spatial indices
    """
    shape = X.shape[1:]
    total = np.prod(shape)
    n_samples = max(1, int(sample_rate * total))
    
    # Compute variances
    variances = compute_tensor_variance(X).flatten()
    variances += 1e-10  # Avoid zeros
    
    # Normalize to probabilities
    probs = variances / np.sum(variances)
    
    # Sample
    sampled = np.random.choice(total, size=n_samples, p=probs, replace=False)
    
    # Convert back to multi-dimensional indices
    return [np.unravel_index(idx, shape) for idx in sampled]


def relative_mse(y_true, y_pred):
    """
    Compute MSE relative to variance.
    
    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
        
    Returns
    -------
    relative_mse : float
        MSE / Var(y_true)
    """
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    
    if var < 1e-10:
        return mse
    
    return mse / var


def print_tree_structure(tree, indent="  "):
    """
    Print tree structure in readable format.
    
    Parameters
    ----------
    tree : TensorDecisionTree
        Fitted tree
    indent : str
        Indentation string
    """
    def _print_node(node, depth=0):
        if node is None:
            return
        
        prefix = indent * depth
        
        if node.is_leaf():
            print(f"{prefix}Leaf: {node.samples_X.shape[0]} samples, "
                  f"mean={node.predicted_value:.3f}")
        else:
            print(f"{prefix}Split on {node.feature_index} <= {node.threshold:.3f}")
            print(f"{prefix}├─ Left:")
            _print_node(node.left, depth + 1)
            print(f"{prefix}└─ Right:")
            _print_node(node.right, depth + 1)
    
    _print_node(tree.root)
