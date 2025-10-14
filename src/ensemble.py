"""
Ensemble methods for tensor decision trees.
"""

import numpy as np
from copy import deepcopy
from .tensor_tree import TensorDecisionTree


class TensorRandomForest:
    """
    Random Forest using tensor slicing for feature selection.
    
    Creates an ensemble of tensor decision trees, each trained on
    a random subset of samples and a random slice of the last dimension.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of each tree
    min_samples_split : int
        Minimum samples to split a node
    split_method : str
        Splitting criterion (passed to TensorDecisionTree)
    split_rank : int
        Rank for tensor decomposition
    slice_size : int, default=4
        Number of slices from last dimension per tree
    sample_rate : float, default=1.0
        Proportion of samples to use per tree (bootstrap)
    n_mode : int
        Number of tensor modes (3 or 4)
    verbose : int, default=0
        Verbosity level
    """
    
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 split_method='variance', split_rank=1, slice_size=4,
                 sample_rate=1.0, n_mode=4, verbose=0):
        
        self.n_estimators = n_estimators
        self.slice_size = slice_size
        self.sample_rate = sample_rate
        
        # Initialize trees
        self.models = []
        for _ in range(n_estimators):
            tree = TensorDecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                split_method=split_method,
                split_rank=split_rank,
                n_mode=n_mode,
                verbose=verbose
            )
            tree.use_mean_as_threshold = False
            self.models.append(tree)
        
        self.partitions = []  # Store slice indices for each tree

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the random forest.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, *spatial_dims)
            Training data (last dimension will be sliced)
        y : ndarray of shape (n_samples,)
            Target values
        X_val : ndarray, optional
            Validation data
        y_val : ndarray, optional
            Validation targets
        """
        n_samples = X.shape[0]
        last_dim_size = X.shape[-1]
        
        self.partitions = []
        
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        for i in range(self.n_estimators):
            # Random slice selection from last dimension
            slice_indices = np.random.choice(
                last_dim_size, 
                size=min(self.slice_size, last_dim_size), 
                replace=False
            )
            self.partitions.append(slice_indices)
            
            # Bootstrap sample selection
            n_bootstrap = int(n_samples * self.sample_rate)
            bootstrap_indices = np.random.choice(n_samples, size=n_bootstrap, replace=True)
            
            # Create training subset
            X_subset = X[bootstrap_indices][..., slice_indices]
            y_subset = y[bootstrap_indices]
            
            # Train tree
            self.models[i].fit(X_subset, y_subset)
            
            # Compute errors
            train_pred = self.models[i].predict(X_subset)
            train_rmse = np.sqrt(np.mean((y_subset - train_pred) ** 2)) / np.std(y_subset)
            
            print(f"Tree {i+1}/{self.n_estimators}: Train RMSE = {train_rmse:.4f}")
            
            # Validation error
            if X_val is not None and y_val is not None:
                X_val_subset = X_val[..., slice_indices]
                val_pred = self.models[i].predict(X_val_subset)
                val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2)) / np.std(y_val)
                print(f"Tree {i+1}/{self.n_estimators}: Val RMSE = {val_rmse:.4f}")
            
            # Forest performance so far
            forest_train_pred = self.predict(X, n_trees=i+1)
            forest_rmse = np.sqrt(np.mean((y - forest_train_pred) ** 2)) / np.std(y)
            print(f"Forest ({i+1} trees): Train RMSE = {forest_rmse:.4f}")
            
            if X_val is not None and y_val is not None:
                forest_val_pred = self.predict(X_val, n_trees=i+1)
                forest_val_rmse = np.sqrt(np.mean((y_val - forest_val_pred) ** 2)) / np.std(y_val)
                print(f"Forest ({i+1} trees): Val RMSE = {forest_val_rmse:.4f}")
            
            print()

    def predict(self, X, regression_method='mean', n_trees=None):
        """
        Predict using ensemble averaging.
        
        Parameters
        ----------
        X : ndarray
            Test data
        regression_method : str, default='mean'
            Prediction method for each tree
        n_trees : int, optional
            Use only first n_trees (for monitoring training)
            
        Returns
        -------
        predictions : ndarray
            Averaged predictions
        """
        trees_to_use = self.models[:n_trees] if n_trees else self.models
        partitions_to_use = self.partitions[:n_trees] if n_trees else self.partitions
        
        predictions = []
        for tree, slice_indices in zip(trees_to_use, partitions_to_use):
            X_subset = X[..., slice_indices]
            pred = tree.predict(X_subset, regression_method)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)


class TensorGradientBoosting:
    """
    Gradient Boosting with tensor decision trees.
    
    Sequentially fits trees to residuals, building an additive model.
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting iterations
    learning_rate : float, default=0.1
        Shrinkage parameter
    max_depth : int, default=3
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples to split
    split_method : str, default='variance'
        Splitting criterion
    split_rank : int, default=1
        Tensor rank
    n_mode : int
        Number of tensor modes
    verbose : int, default=0
        Verbosity level
    """
    
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, split_method='variance', split_rank=1,
                 n_mode=4, verbose=0):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Initialize weak learners
        self.models = []
        for _ in range(n_estimators):
            tree = TensorDecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                split_method=split_method,
                split_rank=split_rank,
                n_mode=n_mode,
                verbose=verbose
            )
            self.models.append(tree)
        
        self.initial_prediction = None

    def fit(self, X, y, X_val=None, y_val=None, regression_method='mean'):
        """
        Fit gradient boosting model.
        
        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Target values
        X_val : ndarray, optional
            Validation data
        y_val : ndarray, optional
            Validation targets
        regression_method : str, default='mean'
            Prediction method for trees
        """
        # Initialize with mean
        self.initial_prediction = np.mean(y)
        current_pred = np.full(y.shape, self.initial_prediction)
        
        print(f"Training Gradient Boosting with {self.n_estimators} iterations...")
        print(f"Initial prediction: {self.initial_prediction:.4f}")
        print()
        
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - current_pred
            
            # Fit tree to residuals
            self.models[i].fit(X, residuals)
            
            # Update predictions
            tree_pred = self.models[i].predict(X, regression_method)
            current_pred += self.learning_rate * tree_pred
            
            # Compute training error
            train_rmse = np.sqrt(np.mean((y - current_pred) ** 2)) / np.std(y)
            print(f"Iteration {i+1}/{self.n_estimators}: Train RMSE = {train_rmse:.4f}")
            
            # Validation error
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val, regression_method, n_estimators=i+1)
                val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2)) / np.std(y_val)
                print(f"Iteration {i+1}/{self.n_estimators}: Val RMSE = {val_rmse:.4f}")
            
            print()

    def predict(self, X, regression_method='mean', n_estimators=None):
        """
        Predict using boosted ensemble.
        
        Parameters
        ----------
        X : ndarray
            Test data
        regression_method : str, default='mean'
            Prediction method
        n_estimators : int, optional
            Use only first n_estimators
            
        Returns
        -------
        predictions : ndarray
            Boosted predictions
        """
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        trees_to_use = self.models[:n_estimators] if n_estimators else self.models
        
        for tree in trees_to_use:
            tree_pred = tree.predict(X, regression_method)
            y_pred += self.learning_rate * tree_pred
        
        return y_pred


class TensorLADBoost:
    """
    LAD (Least Absolute Deviation) TreeBoost.
    
    Gradient boosting variant that optimizes L1 loss (absolute error)
    instead of L2 loss (squared error). More robust to outliers.
    
    Parameters are same as TensorGradientBoosting.
    """
    
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, split_method='variance', split_rank=1,
                 n_mode=4, verbose=0):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        self.models = []
        for _ in range(n_estimators):
            tree = TensorDecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                split_method=split_method,
                split_rank=split_rank,
                n_mode=n_mode,
                verbose=verbose
            )
            self.models.append(tree)
        
        self.initial_prediction = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit LAD boosting model."""
        # Initialize with median (L1 optimal)
        self.initial_prediction = np.median(y)
        current_pred = np.full(y.shape, self.initial_prediction)
        
        print(f"Training LAD Boosting with {self.n_estimators} iterations...")
        print(f"Initial prediction (median): {self.initial_prediction:.4f}")
        print()
        
        for i in range(self.n_estimators):
            # Compute sign of residuals (gradient of L1 loss)
            residuals = np.sign(y - current_pred)
            
            # Fit tree to signed residuals
            self.models[i].fit(X, residuals)
            
            # Update predictions
            tree_pred = self.models[i].predict(X)
            current_pred += self.learning_rate * tree_pred
            
            # Compute L1 loss
            train_mae = np.mean(np.abs(y - current_pred)) / np.std(y)
            print(f"Iteration {i+1}/{self.n_estimators}: Train MAE = {train_mae:.4f}")
            
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val, n_estimators=i+1)
                val_mae = np.mean(np.abs(y_val - val_pred)) / np.std(y_val)
                print(f"Iteration {i+1}/{self.n_estimators}: Val MAE = {val_mae:.4f}")
            
            print()

    def predict(self, X, n_estimators=None):
        """Predict using LAD boosted ensemble."""
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        trees_to_use = self.models[:n_estimators] if n_estimators else self.models
        
        for tree in trees_to_use:
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred
        
        return y_pred
