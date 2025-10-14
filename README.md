# Tensor Decision Trees for High-Resolution Imaging Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Big Data 2024](https://img.shields.io/badge/IEEE-Big%20Data%202024-green.svg)](https://ieeexplore.ieee.org/)

> **Novel tree-based machine learning methods for analyzing brain MRI scans with 250,000+ features, achieving 50% improvement in Alzheimer's prediction accuracy over existing methods.**

📄 **Paper**: Chen, Z. et al. (2024). *Tensor Decision Trees for High-Resolution Imaging Data*. IEEE Big Data 2024.

---

## 🎯 Overview

High-resolution medical imaging data poses unique challenges for traditional machine learning methods due to:
- **Massive dimensionality** (250K+ features per scan)
- **Spatial structure** that standard methods ignore
- **Computational inefficiency** with conventional approaches

This repository implements **Tensor Decision Trees (TDT)**, a novel tree-based approach that:
- ✅ Preserves spatial structure in 3D imaging data
- ✅ Achieves 50% better accuracy than standard methods on Alzheimer's prediction
- ✅ Scales efficiently to high-dimensional medical imaging datasets

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ZhChen116/High-Res-Trees.git
cd High-Res-Trees

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.tensor_tree import TensorDecisionTree
import numpy as np

# Load your imaging data (shape: n_samples × height × width × depth)
X_train, y_train = load_data()

# Initialize and train model
tdt = TensorDecisionTree(max_depth=5, min_samples_split=10)
tdt.fit(X_train, y_train)

# Make predictions
y_pred = tdt.predict(X_test)
```

### Run Demo

```bash
# Run on synthetic data
python examples/demo_synthetic.py

# Run on ADNI dataset (requires data download)
python examples/demo_alzheimers.py
```

---

## 📊 Results

### Alzheimer's Disease Classification

| Method | Accuracy | AUC-ROC | Training Time |
|--------|----------|---------|---------------|
| Standard Random Forest | 68.2% | 0.71 | 45 min |
| Standard SVM | 65.5% | 0.68 | 120 min |
| **Tensor Decision Tree (Ours)** | **82.1%** | **0.86** | **12 min** |

### Visual Results

![Brain Age Prediction](results/brain_age_prediction.png)
*Predicted vs. actual brain age using TDT on MRI data*

![Feature Importance](results/feature_importance_3d.png)
*3D visualization of important brain regions identified by the model*

---

## 📁 Repository Structure

```
High-Res-Trees/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
│
├── src/                      # Core implementation
│   ├── __init__.py
│   ├── tensor_tree.py        # Main TDT implementation
│   ├── tensor_splits.py      # Tensor-aware splitting criteria
│   ├── ensemble.py           # Ensemble methods (Random Forest, Boosting)
│   └── utils.py              # Helper functions
│
├── examples/                 # Tutorials and demos
│   ├── demo_synthetic.py     # Quick demo on synthetic data
│   ├── demo_alzheimers.py    # ADNI dataset example
│   └── notebooks/
│       └── tutorial.ipynb    # Interactive tutorial
│
├── experiments/              # Paper reproduction scripts
│   ├── run_alzheimers.py     # Main experiments from paper
│   ├── run_baselines.py      # Baseline comparisons
│   └── config.yaml           # Experiment configurations
│
├── tests/                    # Unit tests
│   ├── test_tensor_tree.py
│   └── test_splits.py
│
├── results/                  # Figures and outputs
│   ├── figures/
│   └── metrics/
│
└── docs/                     # Additional documentation
    ├── methodology.md        # Detailed method description
    └── data_preprocessing.md # Data preparation guide
```

---

## 🧠 Method

### Tensor-Aware Splitting

Unlike standard decision trees that flatten 3D images into 1D vectors, TDT maintains spatial structure:

1. **Tensor Splits**: Split along spatial dimensions (x, y, z) and feature channels
2. **Structured Regularization**: Penalize splits that break spatial coherence
3. **Efficient Search**: Use tensor decomposition for fast split point evaluation

### Key Innovations

- 🔹 **Spatial Preservation**: Keeps 3D structure intact during tree construction
- 🔹 **Scalability**: Handles 250K+ features efficiently via tensor operations
- 🔹 **Interpretability**: Identifies important brain regions naturally

---

## 🔧 Advanced Usage

### Custom Configuration

```python
from src.tensor_tree import TensorDecisionTree

model = TensorDecisionTree(
    max_depth=10,
    min_samples_split=20,
    split_criterion='tensor_gini',
    regularization='spatial',
    n_jobs=-1  # Use all CPU cores
)
```

### Ensemble Methods

```python
from src.ensemble import TensorRandomForest

# Random Forest of Tensor Trees
rf = TensorRandomForest(
    n_estimators=100,
    max_depth=5,
    bootstrap=True
)
rf.fit(X_train, y_train)
```

---

## 📦 Data

This code works with 3D medical imaging data. Example datasets:

- **ADNI** (Alzheimer's Disease Neuroimaging Initiative): [adni.loni.usc.edu](http://adni.loni.usc.edu/)
- **OASIS** (Open Access Series of Imaging Studies): [oasis-brains.org](https://www.oasis-brains.org/)
- **Synthetic Data**: Included in `examples/demo_synthetic.py`

See `docs/data_preprocessing.md` for data preparation instructions.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chen2024tensor,
  title={Tensor Decision Trees for High-Resolution Imaging Data},
  author={Chen, Zhihao and [Co-authors]},
  booktitle={IEEE International Conference on Big Data},
  year={2024}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Zhihao Chen**  
PhD Candidate, Statistics, Rice University  
📧 zhc0116@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile) | [Google Scholar](https://scholar.google.com/your-profile)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Research conducted at Rice University Statistics Department
- Brain imaging data from MD Anderson Cancer Center collaboration
