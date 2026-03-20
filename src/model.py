import numpy as np


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
        best_feature, best_threshold, best_error = None, None, float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = X[:, feature] > threshold

                if np.sum(left) < 5 or np.sum(right) < 5:
                    continue

                error = (
                    len(y[left]) * mse(y[left]) +
                    len(y[right]) * mse(y[right])
                ) / len(y)

                if error < best_error:
                    best_feature, best_threshold, best_error = feature, threshold, error

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=np.mean(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return TreeNode(value=np.mean(y))

        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold

        return TreeNode(
            feature,
            threshold,
            self.build_tree(X[left], y[left], depth + 1),
            self.build_tree(X[right], y[right], depth + 1),
        )

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        return self.predict_sample(x, node.left if x[node.feature] <= node.threshold else node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


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
        pos_ratio = np.clip(np.mean(y), 1e-5, 1 - 1e-5)  # FIXED EDGE CASE
        self.init_pred = np.log(pos_ratio / (1 - pos_ratio))

        F = np.full(len(y), self.init_pred)

        for _ in range(self.n_estimators):
            probs = self.sigmoid(F)
            residuals = y - probs

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