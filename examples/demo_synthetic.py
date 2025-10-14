"""
Quick demo of Tensor Decision Trees on synthetic 3D data.

This script demonstrates:
1. Creating synthetic tensor data
2. Training a single tensor decision tree
3. Training a random forest
4. Training gradient boosting
5. Comparing performance
"""

import numpy as np
import sys
sys.path.append('..')

from src import (
    TensorDecisionTree,
    TensorRandomForest,
    TensorGradientBoosting,
    relative_mse
)


def generate_synthetic_data(n_samples=200, height=32, width=32, depth=32, noise_level=0.1):
    """
    Generate synthetic 3D tensor data with known structure.
    
    The target is a function of a specific region in the tensor.
    """
    print("Generating synthetic data...")
    print(f"  Shape: ({n_samples}, {height}, {width}, {depth})")
    
    # Generate random 3D tensors
    X = np.random.randn(n_samples, height, width, depth)
    
    # Create target based on specific region
    # y depends on the mean value in a central cube
    region_h = slice(height//3, 2*height//3)
    region_w = slice(width//3, 2*width//3)
    region_d = slice(depth//3, 2*depth//3)
    
    y = X[:, region_h, region_w, region_d].mean(axis=(1, 2, 3))
    
    # Add nonlinearity
    y = np.sin(y * 2) + 0.5 * y**2
    
    # Add noise
    y += noise_level * np.random.randn(n_samples)
    
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    return X, y


def train_single_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate a single tensor decision tree."""
    print("=" * 60)
    print("SINGLE TENSOR DECISION TREE")
    print("=" * 60)
    
    tree = TensorDecisionTree(
        max_depth=5,
        min_samples_split=10,
        split_method='variance',
        split_rank=2,
        n_mode=4,  # 4D tensor (n_samples, h, w, d)
        verbose=0
    )
    
    print("Training tree...")
    tree.fit(X_train, y_train)
    
    print(f"  Tree depth: {tree.get_depth()}")
    print(f"  Number of leaves: {tree.count_leaves()}")
    print()
    
    # Evaluate
    y_train_pred = tree.predict(X_train, regression_method='mean')
    y_test_pred = tree.predict(X_test, regression_method='mean')
    
    train_rmse = relative_mse(y_train, y_train_pred) ** 0.5
    test_rmse = relative_mse(y_test, y_test_pred) ** 0.5
    
    print("Performance (mean regression):")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print()
    
    # Try with CP regression at leaves
    y_train_pred_cp = tree.predict(X_train, regression_method='cp')
    y_test_pred_cp = tree.predict(X_test, regression_method='cp')
    
    train_rmse_cp = relative_mse(y_train, y_train_pred_cp) ** 0.5
    test_rmse_cp = relative_mse(y_test, y_test_pred_cp) ** 0.5
    
    print("Performance (CP regression):")
    print(f"  Train RMSE: {train_rmse_cp:.4f}")
    print(f"  Test RMSE:  {test_rmse_cp:.4f}")
    print()
    
    return tree


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a tensor random forest."""
    print("=" * 60)
    print("TENSOR RANDOM FOREST")
    print("=" * 60)
    
    forest = TensorRandomForest(
        n_estimators=5,
        max_depth=4,
        min_samples_split=10,
        split_method='variance',
        split_rank=2,
        slice_size=4,
        sample_rate=0.8,
        n_mode=4,
        verbose=0
    )
    
    print("Training forest...")
    print()
    forest.fit(X_train, y_train, X_test, y_test)
    
    # Final evaluation
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_rmse = relative_mse(y_train, y_train_pred) ** 0.5
    test_rmse = relative_mse(y_test, y_test_pred) ** 0.5
    
    print("=" * 60)
    print("FINAL FOREST PERFORMANCE:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print()
    
    return forest


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train and evaluate tensor gradient boosting."""
    print("=" * 60)
    print("TENSOR GRADIENT BOOSTING")
    print("=" * 60)
    
    boosting = TensorGradientBoosting(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        split_method='variance',
        split_rank=2,
        n_mode=4,
        verbose=0
    )
    
    boosting.fit(X_train, y_train, X_test, y_test)
    
    # Final evaluation
    y_train_pred = boosting.predict(X_train)
    y_test_pred = boosting.predict(X_test)
    
    train_rmse = relative_mse(y_train, y_train_pred) ** 0.5
    test_rmse = relative_mse(y_test, y_test_pred) ** 0.5
    
    print("=" * 60)
    print("FINAL BOOSTING PERFORMANCE:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print()
    
    return boosting


def main():
    """Run the full demonstration."""
    print("\n" + "=" * 60)
    print("TENSOR DECISION TREE DEMO")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=200, height=16, width=16, depth=16)
    
    # Train/test split
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print()
    
    # Train models
    tree = train_single_tree(X_train, y_train, X_test, y_test)
    forest = train_random_forest(X_train, y_train, X_test, y_test)
    boosting = train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    print("=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("- Tensor trees preserve spatial structure in 3D/4D data")
    print("- Ensemble methods (RF, GBM) improve over single trees")
    print("- CP/Tucker regression at leaves can boost performance")
    print()


if __name__ == "__main__":
    main()
