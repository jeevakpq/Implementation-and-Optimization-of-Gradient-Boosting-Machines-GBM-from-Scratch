import os
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.model import GradientBoostingClassifierScratch


def train():
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingClassifierScratch()
    model.fit(X_train, y_train)

    # ✅ CREATE FOLDER IF NOT EXISTS
    os.makedirs("models", exist_ok=True)

    with open("models/gbm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train()