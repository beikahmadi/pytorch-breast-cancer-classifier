"""Train a small PyTorch classifier on scikit-learn's breast cancer dataset."""
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Repeatable starting weights and data split.
    torch.manual_seed(42)
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # Learn scaling from training data only, then apply it to both sets.
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_function(logits, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | training loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test)).squeeze(1).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=data.target_names, zero_division=0))


if __name__ == "__main__":
    main()
