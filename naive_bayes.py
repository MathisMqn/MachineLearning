import utilities
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.priors = None
        self.means = None
        self.stdevs = None

    def _calculate_likelihoods(self, X):
        num_samples, num_features = X.shape
        likelihoods = np.zeros((num_samples, len(self.priors), num_features))

        for i in range(len(self.priors)):
            class_mean = self.means[i]
            class_std = self.stdevs[i]

            exponent = -0.5 * ((X - class_mean) / class_std) ** 2
            likelihood = 1 / (class_std * np.sqrt(2 * np.pi)) * np.exp(exponent)
            likelihoods[:, i, :] = likelihood

        return np.log(likelihoods)

    def train(self, X, y):
        if not np.all((X >= 0) & (X <= 1)):
            X = utilities.normalize_dataset(X)
            
            self.priors = []
            self.means = []
            self.stdevs = []

            unique_classes = np.unique(y)

            for label in unique_classes:
                indices = np.where(y == label)[0]
                class_data = X[indices]
                self.priors.append(len(indices) / len(y))
                self.means.append(np.mean(class_data, axis=0))
                self.stdevs.append(np.std(class_data, axis=0))

    def predict(self, X):
        if not np.all((X >= 0) & (X <= 1)):
            X = utilities.normalize_dataset(X)
            
            likelihoods = self._calculate_likelihoods(X)
            log_posteriors = np.sum(likelihoods, axis=2) + np.log(self.priors)
            predicted_labels = np.argmax(log_posteriors, axis=1)

        return predicted_labels
