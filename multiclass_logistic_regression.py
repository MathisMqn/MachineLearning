import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class MulticlassLogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.learning_rate = None
        self.loss = []
        self.accuracy = []
        self.final_loss = None
        self.final_accuracy = None

    def _initialize(self):
        self.W = np.random.randn(self.n_classes, self.n_features, 1)
        self.b = np.random.randn(self.n_classes, 1)

    @staticmethod
    def _normalize_dataset(X):
        normalized_dataset = (X - np.min(X)) / (np.max(X) - np.min(X))
    
        return normalized_dataset

    def _softmax(self, X):
        Z = []

        for index, W in enumerate(self.W):
            z = X.dot(W) + self.b[index]
            Z.append(z)

        logits = np.hstack(Z)
        exp_logits = np.exp(logits)
        A = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return A

    def _log_loss(self, A, y):
        m = len(y)
        epsilon = 1e-15

        return -np.sum(y * np.log(A + epsilon)) / m

    def _update_gradients(self, A, X, y):
        m = len(y)
        dZ = A - y
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for i in range(self.n_classes):
            dz = dZ[:, i].reshape((m, 1))
            dW[i] = X.T.dot(dz) / m
            db[i] = np.sum(dz) / m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def display_dataset(self, X, y):
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1],  c="blue", label="Class 0")
            plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1],  c="red", label="Class 1")
            plt.scatter(X[y.flatten() == 2, 0], X[y.flatten() == 2, 1], c="green", label="Class 2")

            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Scatter plot of dataset")
            plt.legend(loc="best")
            plt.show()
        else:
            raise Exception("You can display a dataset only if it contains 2 features")
        
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
        if not np.all((X >= 0) & (X <= 1)):
            X = self._normalize_dataset(X)

        self.learning_rate = learning_rate
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        self._initialize()

        for i in range(n_iters):
            A = self._softmax(X)

            if i % 10 == 0:
                loss_value = self._log_loss(A, y)
                accuracy_value = accuracy_score(y, self.predict(X))
                self.loss.append([i, loss_value])
                self.accuracy.append([i, accuracy_value])

            self._update_gradients(A, X, y)
        
        self.loss, self.accuracy = np.array(self.loss), np.array(self.accuracy)
        self.final_loss, self.final_accuracy = round(self.loss[-1, 1], 2), round(self.accuracy[-1, 1], 2)

    def predict(self, X):
        if self.W is not None:
            if not np.all((X >= 0) & (X <= 1)):
                X = self._normalize_dataset(X)

            A = self._softmax(X)

            return np.argmax(A, axis=1)
        else:
            raise Exception("You must first train the model")
