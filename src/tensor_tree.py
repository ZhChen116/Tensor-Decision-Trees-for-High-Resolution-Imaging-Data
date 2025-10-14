"""
Tensor Decision Tree for high-dimensional imaging data.

This module implements decision trees that preserve spatial structure in 
3D/4D tensor data (e.g., brain MRI scans).
"""

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, constrained_parafac, tucker
from tensorly.regression.cp_regression import CPRegressor
from tensorly.regression.tucker_regression import TuckerRegressor

from .utils import custom_kmeans, L2_norm


class Node:
    """Tree node for tensor decision tree."""
    
    def __init__(self, predicted_value, leaf_index, split_method=None, 
                 samples_X=None, samples_y=None, cp_model=None, tucker_model=None):
        self.predicted_value = predicted_value
        self.feature_index = None
        self.threshold = None
        self.parent = None
        self.left = None
        self.right = None
        self.split_method = split_method
        self.samples_X = samples_X
        self.samples_y = samples_y
        self.cp_model = cp_model
        self.tucker_model = tucker_model
        self.split_loss = None
        self.leaf_index = leaf_index

    def is_leaf(self):
        """Check if node is a leaf."""
        return (self.left is None) and (self.right is None)
    
    def get_leaves(self):
        """Get all leaf nodes in subtree."""
        if self.is_leaf():
            return [self]
        
        leaves = []
        if self.left is not None:
            leaves.extend(self.left.get_leaves())
        if self.right is not None:
            leaves.extend(self.right.get_leaves())
        return leaves

    def get_depth(self):
        """Get depth of this node."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1


class TensorDecisionTree:
    """
    Decision tree for tensor-structured data.
    
    Handles 3D or 4D tensors (e.g., medical imaging) while preserving
    spatial structure during splitting.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    split_method : str, default='variance'
        Splitting criterion. Options:
        - 'variance': Minimize variance
        - 'variance_LS': Variance with leverage score sampling
        - 'lowrank': Low-rank tensor approximation error
        - 'lowrank_reg': Regression-based low-rank splitting
        - 'kmeans': K-means clustering of tensor slices
    split_rank : int, default=1
        Rank for low-rank decomposition
    n_mode : int, required
        Number of modes in tensor (3 or 4)
    verbose : int, default=0
        Verbosity level
    const_array : array-like, optional
        Indices to exclude from splitting (constrained features)
        
    Attributes
    ----------
    root : Node
        Root node of the tree
    """
    
    def __init__(self, max_depth=5, min_samples_split=2, split_method='variance',
                 split_rank=1, CP_reg_rank=None, Tucker_reg_rank=None, 
                 n_mode=None, verbose=0, const_array=None):
        
        if n_mode is None:
            raise ValueError("n_mode must be specified (3 or 4)")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.split_method = split_method
        self.split_rank = split_rank
        self.n_mode = n_mode
        self.verbose = verbose
        self.const_array = const_array
        
        # Regression ranks
        self.CP_reg_rank = CP_reg_rank if CP_reg_rank is not None else split_rank
        self.Tucker_reg_rank = Tucker_reg_rank if Tucker_reg_rank is not None else split_rank
        
        # Tree state
        self.root = None
        self.leaf_counter = 0
        
        # Splitting parameters
        self.use_mean_as_threshold = True
        self.lowrank_method = 'cp'
        self.sample_rate = 0.01  # For leverage score sampling
        self.tolerance = 0.1  # For branch-and-bound
        self.modelmaxiter = 200
        
        # Classifier for kmeans method
        self.classifier = None

    def fit(self, X, y):
        """
        Build the tensor decision tree.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, *spatial_dims)
            Training tensor data
        y : ndarray of shape (n_samples,)
            Target values
        """
        if X.ndim == 3:
            self.n_mode = 3
            self.mode1, self.mode2, self.mode3 = X.shape
        elif X.ndim == 4:
            self.n_mode = 4
            self.mode1, self.mode2, self.mode3, self.mode4 = X.shape
        else:
            raise ValueError("X must be 3D or 4D tensor")
        
        self.leaf_counter = 0
        self.root = self._build_tree(X, y)

    def predict(self, X, regression_method='mean'):
        """
        Predict target values.
        
        Parameters
        ----------
        X : ndarray
            Test tensor data
        regression_method : str, default='mean'
            Prediction method: 'mean', 'cp', or 'tucker'
            
        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        if regression_method == 'mean':
            return np.array([self._traverse_tree(x, self.root) for x in X])
        elif regression_method == 'cp':
            return np.array([self._traverse_tree_with_cp(x, self.root, 0) for x in X])
        elif regression_method == 'tucker':
            return np.array([self._traverse_tree_with_tucker(x, self.root, 0) for x in X])
        else:
            raise ValueError(f"Unknown regression method: {regression_method}")

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        # Stopping criteria
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return self._create_leaf_node(X, y, depth)
        
        # Find best split
        feature_index, threshold, loss = self._get_best_split(X, y, depth)
        
        if feature_index is None or threshold is None:
            return self._create_leaf_node(X, y, depth)
        
        # Create internal node
        node = Node(predicted_value=None, leaf_index=self.leaf_counter,
                   samples_X=X, samples_y=y)
        self.leaf_counter += 1
        
        node.feature_index = feature_index
        node.threshold = threshold
        node.split_loss = loss
        
        # Split data and recurse
        left_idx, right_idx = self._split(X, y, feature_index, threshold)
        
        if np.sum(left_idx) >= self.min_samples_split:
            node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
            if node.left is not None:
                node.left.parent = node
                
        if np.sum(right_idx) >= self.min_samples_split:
            node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
            if node.right is not None:
                node.right.parent = node
        
        # If split failed, make this a leaf
        if node.left is None and node.right is None:
            return self._create_leaf_node(X, y, depth)
        
        return node

    def _create_leaf_node(self, X, y, depth):
        """Create a leaf node with regression models."""
        # Train CP model
        cp_model = CPRegressor(weight_rank=self.CP_reg_rank, verbose=self.verbose)
        cp_model.fit(X, y)
        
        # Train Tucker model
        if self.n_mode == 3:
            tucker_ranks = [self.Tucker_reg_rank, self.Tucker_reg_rank]
        else:
            tucker_ranks = [self.Tucker_reg_rank, self.Tucker_reg_rank, self.Tucker_reg_rank]
        
        tucker_model = TuckerRegressor(
            weight_ranks=tucker_ranks, 
            tol=1e-7, 
            n_iter_max=self.modelmaxiter, 
            reg_W=1, 
            verbose=self.verbose
        )
        tucker_model.fit(X, y)
        
        self.leaf_counter += 1
        return Node(
            predicted_value=np.mean(y),
            leaf_index=self.leaf_counter,
            samples_X=X,
            samples_y=y,
            cp_model=cp_model,
            tucker_model=tucker_model
        )

    def _get_best_split(self, X, y, depth):
        """Find the best feature and threshold to split on."""
        if self.split_method == 'kmeans':
            return self._split(X, y, [], []), None, None
        
        # Map split method to error type and optimization
        method_map = {
            'variance': ('variance', 'exhaustive'),
            'variance_LS': ('variance', 'LS'),
            'lowrank': ('low_rank', 'exhaustive'),
            'lowrank_LS': ('low_rank', 'LS'),
            'lowrank_reg': ('low_rank_reg', 'exhaustive'),
            'lowrank_reg_LS': ('low_rank_reg', 'LS'),
        }
        
        if self.split_method not in method_map:
            raise ValueError(f"Unknown split method: {self.split_method}")
        
        error_type, optimization = method_map[self.split_method]
        
        # Error function
        def error_fn(subset, subset_y):
            if error_type == 'variance':
                return np.var(subset_y) * len(subset_y)
            elif error_type == 'low_rank':
                return self._rank_k_approx_error(subset)
            elif error_type == 'low_rank_reg':
                return self._rank_k_reg_error(subset, subset_y)
        
        # Get candidate indices based on optimization strategy
        indices = self._get_candidate_indices(X, optimization)
        
        # Find best split
        best_err = np.inf
        best_feature_index = None
        best_threshold = None
        
        for feature_index in indices:
            feature_values = X[(slice(None),) + feature_index]
            threshold = np.mean(feature_values)
            
            left_idx = feature_values <= threshold
            right_idx = ~left_idx
            
            if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                continue
            
            err = error_fn(X[left_idx], y[left_idx]) + error_fn(X[right_idx], y[right_idx])
            
            if err < best_err:
                best_err = err
                best_feature_index = feature_index
                best_threshold = threshold
        
        return best_feature_index, best_threshold, best_err

    def _get_candidate_indices(self, X, optimization):
        """Get candidate feature indices based on optimization strategy."""
        if optimization == 'exhaustive':
            if self.const_array is not None:
                const_set = set(self.const_array)
                all_indices = list(np.ndindex(X.shape[1:]))
                return [idx for idx in all_indices if idx not in const_set]
            return np.ndindex(X.shape[1:])
        
        elif optimization == 'LS':
            # Leverage score sampling
            shape = X.shape[1:]
            total = np.prod(shape)
            n_samples = max(1, int(self.sample_rate * total))
            
            # Compute variances for sampling
            variances = np.array([np.var(X[(slice(None),) + idx]) 
                                 for idx in np.ndindex(shape)])
            variances += 1e-10  # Avoid zeros
            probs = variances / np.sum(variances)
            
            sampled = np.random.choice(total, size=n_samples, p=probs, replace=False)
            return [np.unravel_index(idx, shape) for idx in sampled]

    def _split(self, X, y, feature_index, threshold):
        """Split data based on feature and threshold."""
        if self.split_method == 'kmeans':
            # K-means clustering
            if X.ndim == 3:
                X_list = [X[i] for i in range(X.shape[0])]
            else:
                X_list = [X[i] for i in range(X.shape[0])]
            
            self.classifier = custom_kmeans(2, L2_norm)
            self.classifier.fit(X_list)
            labels = self.classifier.predict(X_list)
            
            left_idx = labels == 0
            right_idx = ~left_idx
        else:
            # Threshold-based splitting
            if X.ndim == 3:
                left_idx = X[:, feature_index[0], feature_index[1]] <= threshold
            else:
                left_idx = X[:, feature_index[0], feature_index[1], feature_index[2]] <= threshold
            right_idx = ~left_idx
        
        return left_idx, right_idx

    def _rank_k_approx_error(self, X):
        """Compute low-rank approximation error."""
        if len(X) <= self.min_samples_split:
            return np.inf
        
        if self.lowrank_method == 'cp':
            weights, factors = parafac(X, rank=self.split_rank, 
                                      l2_reg=np.finfo(np.float32).eps)
            approx = tl.cp_to_tensor((weights, factors))
        elif self.lowrank_method == 'tucker':
            core, factors = tucker(X, rank=[X.shape[0], self.split_rank, self.split_rank])
            approx = tl.tucker_to_tensor((core, factors))
        else:
            raise ValueError(f"Unknown lowrank method: {self.lowrank_method}")
        
        return tl.norm(X - approx)

    def _rank_k_reg_error(self, X, y):
        """Compute regression-based low-rank error."""
        if len(X) <= self.min_samples_split:
            return np.inf
        
        if self.lowrank_method == 'cp':
            model = CPRegressor(weight_rank=self.CP_reg_rank, verbose=self.verbose)
        elif self.lowrank_method == 'tucker':
            ranks = [self.Tucker_reg_rank] * (self.n_mode - 1)
            model = TuckerRegressor(weight_ranks=ranks, tol=1e-7, 
                                   n_iter_max=self.modelmaxiter, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown lowrank method: {self.lowrank_method}")
        
        model.fit(X, y)
        y_pred = model.predict(X)
        return tl.norm(y - y_pred)

    def _traverse_tree(self, x, node):
        """Traverse tree for prediction."""
        if node is None or node.is_leaf():
            return np.mean(node.samples_y) if node is not None else 0.0
        
        # Check split condition
        if self.split_method == 'kmeans':
            label = self.classifier.predict([x])[0]
            next_node = node.left if label == 0 else node.right
        else:
            if x.ndim == 2:  # 3D data
                value = x[node.feature_index[0], node.feature_index[1]]
            else:  # 4D data
                value = x[node.feature_index[0], node.feature_index[1], node.feature_index[2]]
            
            next_node = node.left if value <= node.threshold else node.right
        
        return self._traverse_tree(x, next_node)

    def _traverse_tree_with_cp(self, x, node, depth):
        """Traverse tree using CP regression at leaves."""
        if node is None:
            return np.nan
        
        if node.is_leaf():
            if node.cp_model is None:
                cp_model = CPRegressor(weight_rank=self.CP_reg_rank, verbose=self.verbose)
                cp_model.fit(node.samples_X, node.samples_y)
                node.cp_model = cp_model
            return node.cp_model.predict(x.reshape(1, *x.shape))[0]
        
        # Navigate to appropriate child
        if self.split_method == 'kmeans':
            label = self.classifier.predict([x])[0]
            next_node = node.left if label == 0 else node.right
        else:
            if x.ndim == 2:
                value = x[node.feature_index[0], node.feature_index[1]]
            else:
                value = x[node.feature_index[0], node.feature_index[1], node.feature_index[2]]
            next_node = node.left if value <= node.threshold else node.right
        
        return self._traverse_tree_with_cp(x, next_node, depth + 1)

    def _traverse_tree_with_tucker(self, x, node, depth):
        """Traverse tree using Tucker regression at leaves."""
        if node is None:
            return np.nan
        
        if node.is_leaf():
            if node.tucker_model is None:
                ranks = [self.Tucker_reg_rank] * (self.n_mode - 1)
                tucker_model = TuckerRegressor(weight_ranks=ranks, tol=1e-7,
                                              n_iter_max=self.modelmaxiter, 
                                              reg_W=1, verbose=self.verbose)
                tucker_model.fit(node.samples_X, node.samples_y)
                node.tucker_model = tucker_model
            return node.tucker_model.predict(x.reshape(1, *x.shape))[0]
        
        # Navigate to appropriate child
        if self.split_method == 'kmeans':
            label = self.classifier.predict([x])[0]
            next_node = node.left if label == 0 else node.right
        else:
            if x.ndim == 2:
                value = x[node.feature_index[0], node.feature_index[1]]
            else:
                value = x[node.feature_index[0], node.feature_index[1], node.feature_index[2]]
            next_node = node.left if value <= node.threshold else node.right
        
        return self._traverse_tree_with_tucker(x, next_node, depth + 1)

    def get_depth(self):
        """Get maximum depth of the tree."""
        def _get_depth(node):
            if node is None or node.is_leaf():
                return 0
            left_depth = _get_depth(node.left)
            right_depth = _get_depth(node.right)
            return 1 + max(left_depth, right_depth)
        
        return _get_depth(self.root)

    def count_leaves(self):
        """Count number of leaf nodes."""
        if self.root is None:
            return 0
        return len(self.root.get_leaves())
