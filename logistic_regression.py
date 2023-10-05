import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.learning_rate = None
        self.loss = []
        self.accuracy = []
        self.final_loss = None
        self.final_accuracy = None

    def _initialize(self, n_features):
        self.W = np.random.randn(n_features, 1)
        self.b = np.random.randn(1)

    def _sigmoid(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))

        return A

    def _log_loss(self, A, y):
        m = len(y)
        epsilon = 1e-15

        return -np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon)) / m

    def _update_gradients(self, A, X, y):
        m = len(y)
        dW = X.T.dot(A - y) / m
        db = np.sum(A - y) / m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def display_dataset(self, X, y):
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1],  c="blue", label="Class 0")
            plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1],  c="red", label="Class 1")

            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Scatter plot of dataset")
            plt.legend(loc="best")
            plt.show()
        else:
            raise Exception("You can display a dataset only if it contains 2 features")

    def display_decision_boundary(self, X, y):
        if self.W is not None:
            if X.shape[1] == 2:
                plt.figure(figsize=(8, 6))

                plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1],  c="blue", label="Class 0")
                plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1],  c="red", label="Class 1")

                x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
                x1 = np.linspace(x1_min, x1_max, 400)
                x2 = (-self.W[0] * x1 - self.b) / self.W[1]
                plt.plot(x1, x2, c="black", lw=1, label="Decision boundary")

                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.title("Scatter plot of dataset")
                plt.legend(loc="best")

                plt.show()
            else:
                raise Exception("You can display a decision boundary only if the dataset contains 2 features")
        else:
            raise Exception("You must first train the model")

    def display_roc_curve(self, X, y_true):
        if self.W is not None:
            roc_points = []
            thresholds = np.arange(0.1, 1.0, 0.1)

            for threshold in thresholds:
                y_pred = self.predict(X, threshold=threshold)
                true_positives = np.sum(np.logical_and(y_pred == True, y_true == True))
                false_positives = np.sum(np.logical_and(y_pred == True, y_true == False))
                true_negatives = np.sum(np.logical_and(y_pred == False, y_true == False))
                false_negatives = np.sum(np.logical_and(y_pred == False, y_true == True))

                false_positive_rate = false_positives / (false_positives + true_negatives)
                true_positive_rate = true_positives / (true_positives + false_negatives)

                roc_points.append((false_positive_rate, true_positive_rate))

            plt.figure(figsize=(8, 6))

            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.plot([p[0] for p in roc_points], [p[1] for p in roc_points], marker="o", linestyle="-")

            for i, threshold in enumerate(thresholds):
                plt.annotate(f"{threshold: .1f}", (roc_points[i][0], roc_points[i][1]))

            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.grid()
            plt.show()
        else:
            raise Exception("You must first train the model")
        
    def display_learning_curves(self):
        if self.W is not None:
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.plot(self.loss[:, 0], self.loss[:, 1], label="Log loss")
            plt.xlabel("Iterations")
            plt.ylabel("Log loss")
            plt.title(f"Log loss after learning: {self.final_loss}")
            plt.legend(loc="best")

            plt.subplot(1, 2, 2)
            plt.plot(self.accuracy[:, 0], self.accuracy[:, 1], label="Accuracy")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy after learning: {self.final_accuracy}")
            plt.legend(loc="best")

            plt.show()
        else:
            raise Exception("You must first train the model")

    def train(self, X, y, n_iters=1000, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._initialize(X.shape[1])

        for i in range(n_iters):
            A = self._sigmoid(X)

            if i % 10 == 0:
                loss_value = self._log_loss(A, y)
                accuracy_value = accuracy_score(y, self.predict(X))
                self.loss.append([i, loss_value])
                self.accuracy.append([i, accuracy_value])

            self._update_gradients(A, X, y)

        self.loss, self.accuracy = np.array(self.loss), np.array(self.accuracy)
        self.final_loss, self.final_accuracy = round(self.loss[-1, 1], 2), round(self.accuracy[-1, 1], 2)

    def predict(self, X, threshold=0.5):
        if self.W is not None:
            A = self._sigmoid(X)

            return A >= threshold
        else:
            raise Exception("You must first train the model")
