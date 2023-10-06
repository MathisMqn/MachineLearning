import utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.loss_history = []
        self.accuracy_history = []
        self.final_loss = None
        self.final_accuracy = None

    def _initialize_weights(self, n_features):
        self.weights = np.random.randn(n_features, 1)
        self.bias = np.random.randn(1)

    def _calculate_sigmoid(self, X):
        linear_combination = X.dot(self.weights) + self.bias
        sigmoid_result = 1 / (1 + np.exp(-linear_combination))

        return sigmoid_result

    def _get_log_loss(self, sigmoid_result, labels):
        m = len(labels)
        epsilon = 1e-15
        loss = -np.sum(labels * np.log(sigmoid_result + epsilon) + (1 - labels) * np.log(1 - sigmoid_result + epsilon)) / m

        return loss

    def _update_gradients(self, sigmoid_result, X, labels):
        m = len(labels)
        dW = X.T.dot(sigmoid_result - labels) / m
        db = np.sum(sigmoid_result - labels) / m

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db

    def display_decision_boundary(self, X, y):
        if self.weights is not None:
            if X.shape[1] == 2:
                if not np.all((X >= 0) & (X <= 1)):
                    X = utilities.normalize_dataset(X)
                    
                plt.figure(figsize=(8, 6))

                plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], label="Class 0")
                plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], label="Class 1")

                x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
                x1 = np.linspace(x1_min, x1_max, 400)
                x2 = (-self.weights[0] * x1 - self.bias) / self.weights[1]
                plt.plot(x1, x2, lw=1, label="Decision boundary")

                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.title("Scatter plot of dataset")
                plt.legend(loc="best")

                plt.show()
            else:
                raise Exception("You can display a decision boundary only if the dataset contains 2 features")
        else:
            raise Exception("You must first train the model")

    def display_roc_curve(self, X, y):
        if self.weights is not None:
            if not np.all((X >= 0) & (X <= 1)):
                X = utilities.normalize_dataset(X)

            roc_points = []
            thresholds = np.arange(0.1, 1.0, 0.1)

            for threshold in thresholds:
                y_pred = self.predict(X, threshold=threshold)
                true_positives = np.sum(np.logical_and(y_pred == True, y == True))
                false_positives = np.sum(np.logical_and(y_pred == True, y == False))
                true_negatives = np.sum(np.logical_and(y_pred == False, y == False))
                false_negatives = np.sum(np.logical_and(y_pred == False, y == True))

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
        if self.weights is not None:
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history[:, 0], self.loss_history[:, 1], label="Log loss")
            plt.xlabel("Iterations")
            plt.ylabel("Log loss")
            plt.title(f"Log loss after learning: {self.final_loss}")
            plt.legend(loc="best")

            plt.subplot(1, 2, 2)
            plt.plot(self.accuracy_history[:, 0], self.accuracy_history[:, 1], label="Accuracy")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy after learning: {self.final_accuracy}")
            plt.legend(loc="best")

            plt.show()
        else:
            raise Exception("You must first train the model")

    def train(self, X, y, n_iters=1000, learning_rate=0.1):
        if not np.all((X >= 0) & (X <= 1)):
            X = utilities.normalize_dataset(X)

        self.learning_rate = learning_rate
        self._initialize_weights(X.shape[1])

        for i in range(n_iters):
            sigmoid_result = self._calculate_sigmoid(X)

            if i % 10 == 0:
                loss = self._get_log_loss(sigmoid_result, y)
                accuracy = accuracy_score(y, self.predict(X))
                self.loss_history.append([i, loss])
                self.accuracy_history.append([i, accuracy])

            self._update_gradients(sigmoid_result, X, y)

        self.loss_history, self.accuracy_history = np.array(self.loss_history), np.array(self.accuracy_history)
        self.final_loss, self.final_accuracy = round(self.loss_history[-1, 1], 2), round(self.accuracy_history[-1, 1], 2)

    def predict(self, X, threshold=0.5):
        if self.weights is not None:
            if not np.all((X >= 0) & (X <= 1)):
                X = utilities.normalize_dataset(X)

            probs = self._calculate_sigmoid(X)

            return probs >= threshold
        else:
            raise Exception("You must first train the model")
