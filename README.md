# Breast Cancer Classification with PyTorch

A beginner-friendly machine learning project that trains a neural network to classify samples as malignant or benign using the built-in Wisconsin breast cancer dataset in scikit-learn.

**Educational use only. This model is not a medical device and must not be used for diagnosis.** The dataset is small and does not establish performance in clinical practice.

## What the project does

1. Loads 569 samples with 30 numerical features each.
2. Splits the data into training (80%) and test (20%) sets, preserving class proportions.
3. Standardizes features using statistics from the training set only.
4. Trains a small neural network using PyTorch.
5. Reports accuracy, a confusion matrix, precision, recall and F1 score on the test set.

The target labels are 0 for malignant and 1 for benign. The model produces a probability for class 1; predictions use a 0.5 threshold. Accuracy alone can hide important errors, so examine the confusion matrix and recall as well.

## Run locally

Install Python 3.10 or newer, then open a terminal in this folder:

```bash
python -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, or on Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then run:

```bash
pip install -r requirements.txt
python train.py
```

## Understand the code

- `nn.Linear`: a layer that learns weights.
- `nn.ReLU`: lets the network learn non-linear relationships.
- `BCEWithLogitsLoss`: measures error for binary classification.
- `loss.backward()`: calculates how each weight contributes to error.
- `optimizer.step()`: updates the weights.

A fixed seed makes the experiment more repeatable, although exact results can vary across software and hardware versions. This is a simple baseline; a single test split does not prove the model will generalize.

## Ideas for your next version

Compare with logistic regression, use cross-validation, explore feature importance, and document limitations. Keep the test set separate from model selection.

## Dataset

Scikit-learn includes the Breast Cancer Wisconsin (Diagnostic) dataset. See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html for the dataset description and original references.
