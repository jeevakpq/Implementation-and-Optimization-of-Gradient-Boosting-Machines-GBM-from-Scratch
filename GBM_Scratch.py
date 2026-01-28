"""
Project Title:
Implementation and Optimization of Gradient Boosting Machines (GBM) from Scratch

Author: Student
Description:
This script implements a Gradient Boosting Machine (GBM) classifier
from scratch using NumPy, including CART regression trees, pseudo-residuals,
shrinkage, hyperparameter tuning, and comparison with XGBoost.
"""

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


# =========================================================
# REGRESSION TREE IMPLEMENTATION (CART)
# =========================================================

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def mse(y):
    return np.mean((y - np.mean(y)) ** 2)


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_error = float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                error = (
                    len(left_y) * mse(left_y) +
                    len(right_y) * mse(right_y)
                ) / len(y)

                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=np.mean(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature, threshold, left_child, right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


# =========================================================
# GRADIENT BOOSTING CLASSIFIER (FROM SCRATCH)
# =========================================================

class GradientBoostingClassifierScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        pos_ratio = np.mean(y)
        self.init_pred = np.log(pos_ratio / (1 - pos_ratio))

        F = np.full(len(y), self.init_pred)

        for _ in range(self.n_estimators):
            probabilities = self.sigmoid(F)
            residuals = y - probabilities

            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict_proba(self, X):
        F = np.full(len(X), self.init_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return self.sigmoid(F)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    # -------------------------------
    # Dataset Generation
    # -------------------------------
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Dataset Shape:", X.shape)

    # -------------------------------
    # Train Custom GBM
    # -------------------------------
    start = time.time()

    gbm = GradientBoostingClassifierScratch(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )

    gbm.fit(X_train, y_train)
    custom_time = time.time() - start

    y_pred = gbm.predict(X_test)
    y_prob = gbm.predict_proba(X_test)

    print("\nCustom GBM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Training Time:", custom_time)

    # -------------------------------
    # Hyperparameter Tuning
    # -------------------------------
    print("\nHyperparameter Tuning Results")
    for lr in [0.05, 0.1]:
        for n in [50, 100]:
            for depth in [2, 3]:
                model = GradientBoostingClassifierScratch(
                    n_estimators=n,
                    learning_rate=lr,
                    max_depth=depth
                )
                model.fit(X_train, y_train)
                auc = roc_auc_score(y_test, model.predict_proba(X_test))
                print(f"LR={lr}, Trees={n}, Depth={depth}, AUC={auc:.4f}")

    # -------------------------------
    # XGBoost Comparison
    # -------------------------------
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric="logloss",
        use_label_encoder=False
    )

    start = time.time()
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start

    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)

    print("\nXGBoost Results")
    print("Accuracy:", accuracy_score(y_test, xgb_pred))
    print("AUC:", roc_auc_score(y_test, xgb_prob))
    print("Training Time:", xgb_time)

    # -------------------------------
    # Final Comparison
    # -------------------------------
    print("\nFinal Comparison Summary")
    print("--------------------------------")
    print("Custom GBM | AUC:", roc_auc_score(y_test, y_prob), "| Time:", custom_time)
    print("XGBoost   | AUC:", roc_auc_score(y_test, xgb_prob), "| Time:", xgb_time)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
