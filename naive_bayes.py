import numpy as np
import utilities


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


class NaiveBayesClassifier:
    """
    Naive Bayes classifier.

    Attributes
    ----------
    priors : list
        Priors of each class
    means : list
        Means of each class
    stdevs : list
        Standard deviations of each class
    """

    def __init__(self):
        self.priors = None
        self.means = None
        self.stdevs = None

    def _calculate_likelihoods(self, features: np.ndarray) -> np.ndarray:
        """
        Calculates the likelihoods of each class for the input data.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset

        Returns
        -------
        np.ndarray
            Likelihoods for each class
        """

        num_samples, num_features = features.shape
        likelihoods = np.zeros((num_samples, len(self.priors), num_features))

        for i in range(len(self.priors)):
            class_mean = self.means[i]
            class_std = self.stdevs[i]

            exponent = -0.5 * ((features - class_mean) / class_std) ** 2
            likelihood = 1 / (class_std * np.sqrt(2 * np.pi)) * np.exp(exponent)
            likelihoods[:, i, :] = likelihood

        return np.log(likelihoods)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains the Naive Bayes model on the input data.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset
        labels : np.ndarray
            Labels of the dataset
        """

        if not np.all((features >= 0) & (features <= 1)):
            features = utilities.normalize_dataset(features)

        self.priors = []
        self.means = []
        self.stdevs = []

        unique_labels = np.unique(labels)

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            class_data = features[indices]
            self.priors.append(len(indices) / len(labels))
            self.means.append(np.mean(class_data, axis=0))
            self.stdevs.append(np.std(class_data, axis=0))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset

        Returns
        -------
        np.ndarray
            Predicted labels
        """

        if len(self.priors) > 0:
            if not np.all((features >= 0) & (features <= 1)):
                features = utilities.normalize_dataset(features)

            likelihoods = self._calculate_likelihoods(features)
            log_posteriors = np.sum(likelihoods, axis=2) + np.log(self.priors)
            predicted_labels = np.argmax(log_posteriors, axis=1)

            return predicted_labels

        raise ModelNotTrainedError()
