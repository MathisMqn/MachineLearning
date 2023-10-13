import numpy as np
import matplotlib.pyplot as plt


class NearestNeighborsClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def _calculate_euclidean_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
    
    def display_roc_curve(self, X, y):
        if self.X_train is not None:
            K_list = np.arange(1, 11, 2)
            roc_points = np.zeros((len(K_list), 2))

            for index, K in enumerate(K_list):
                y_pred = self.predict(X, K=K)
                true_positives = np.sum(np.logical_and(y_pred == True, y == True))
                false_positives = np.sum(np.logical_and(y_pred == True, y == False))
                true_negatives = np.sum(np.logical_and(y_pred == False, y == False))
                false_negatives = np.sum(np.logical_and(y_pred == False, y == True))

                false_positive_rate = false_positives / (false_positives + true_negatives)
                true_positive_rate = true_positives / (true_positives + false_negatives)

                roc_points[index] = false_positive_rate, true_positive_rate

            plt.figure(figsize=(8, 6))

            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.scatter(roc_points[:, 0], roc_points[:, 1], marker="o", linestyle="-")

            for i, K in enumerate(K_list):
                plt.annotate(f"{K: .0f}", (roc_points[i][0], roc_points[i][1]))

            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.grid()
            plt.show()
        else:
            raise Exception("You must first train the model")

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, K=5):
        if K % 2 == 1:
            if self.X_train is not None:
                y_pred = np.zeros((X.shape[0], 1))

                for i in range(X.shape[0]):
                    distances = self._calculate_euclidean_distance(X[i])
                    nearest_neighbor = self.y_train[np.argpartition(distances, K)[:K]].flatten()
                    y_pred[i] = np.bincount(nearest_neighbor).argmax()

                return y_pred
            else:
                raise Exception("You must first train the model")
        else:
            raise Exception("K must be an odd number")
