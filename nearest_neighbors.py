import numpy as np
import matplotlib.pyplot as plt


class ModelNotTrainedError(Exception):
    """
    Exception raised when the model has not been trained.

    Attributes
    ----------
    message : str
        Error message
    """

    def __init__(self, message="You must first train the model"):
        self.message = message
        super().__init__(self.message)

class NearestNeighborsClassifier:
    """
    Nearest Neighbors classifier.

    Attributes
    ----------
    features_train : np.ndarray
        Features of the training set
    labels_train : np.ndarray
        Labels of the training set
    """

    def __init__(self):
        self.features_train = None
        self.labels_train = None

    def _calculate_euclidean_distance(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((self.features_train - x) ** 2, axis=1))

    def display_roc_curve(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Displays the ROC curve.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset
        labels : np.ndarray
            Labels of the dataset
        """

        if self.features_train is not None:
            k_list = np.arange(1, 11, 2)
            roc_points = np.zeros((len(k_list), 2))

            for index, k in enumerate(k_list):
                predicted_values = self.predict(features, k=k)
                true_positives = np.sum(np.logical_and(predicted_values == True, labels == True))
                false_positives = np.sum(np.logical_and(predicted_values == True, labels == False))
                true_negatives = np.sum(np.logical_and(predicted_values == False, labels == False))
                false_negatives = np.sum(np.logical_and(predicted_values == False, labels == True))

                false_positive_rate = false_positives / (false_positives + true_negatives)
                true_positive_rate = true_positives / (true_positives + false_negatives)

                roc_points[index] = false_positive_rate, true_positive_rate

            plt.figure(figsize=(8, 6))

            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.scatter(roc_points[:, 0], roc_points[:, 1], marker="o", linestyle="-")

            for i, k in enumerate(k_list):
                plt.annotate(f"{k: .0f}", (roc_points[i][0], roc_points[i][1]))

            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.grid()
            plt.show()
        else:
            raise ModelNotTrainedError()

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains the model on the input data.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset
        labels : np.ndarray
            Labels of the dataset
        """

        self.features_train = features
        self.labels_train = labels

    def predict(self, features: np.ndarray, k: int=5) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset
        k : int
            Number of neighbors to consider

        Returns
        -------
        np.ndarray
            Predicted labels
        """

        if k % 2 == 1:
            if self.labels_train is not None:
                predicted_values = np.zeros((features.shape[0], 1))

                for i in range(features.shape[0]):
                    distances = self._calculate_euclidean_distance(features[i])
                    nearest_neighbor = self.labels_train[np.argpartition(distances, k)[:k]].flatten()
                    predicted_values[i] = np.bincount(nearest_neighbor).argmax()

                return predicted_values

            raise ModelNotTrainedError()

        raise ValueError("k must be an odd number")
