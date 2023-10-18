import numpy as np
import pandas as pd


def get_confusion_matrix(predicted_values: np.ndarray, true_values: np.ndarray) -> np.ndarray:
    """
    Calculates the confusion matrix of a classification task.

    Parameters
    ----------
    predicted_values : np.ndarray
        Predicted values of the classification task
    true_values : np.ndarray
        True values of the classification task

    Returns
    -------
    np.ndarray
        Confusion matrix
    """

    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(predicted_values, true_values)

    return matrix

def get_accuracy_score(predicted_values: np.ndarray, true_values: np.ndarray) -> float:
    """
    Calculates the accuracy score of a classification task.

    Parameters
    ----------
    predicted_values : np.ndarray
        Predicted values of the classification task
    true_values : np.ndarray
        True values of the classification task

    Returns
    -------
    float
        Accuracy score
    """

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(predicted_values, true_values)

    return accuracy


class Dataset:
    """
    Loads a dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file
    delimiter : str
        Delimiter used in the CSV file. If not specified, it will be automatically detected

    Attributes
    ----------
    path : str
        Path to the CSV file
    delimiter : str
        Delimiter used in the CSV file
    dataset : pd.DataFrame
        Dataset loaded from the CSV file
    """

    def __init__(self, path: str, delimiter: str=None):
        self.path = path
        self.delimiter = delimiter
        self.dataset = self._load()

    @staticmethod
    def normalize(features: np.ndarray) -> np.ndarray:
        """
        Normalizes a dataset using the min-max normalization.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset

        Returns
        -------
        np.ndarray
            Normalized dataset
        """

        dataset = (features - np.min(features)) / (np.max(features) - np.min(features))

        return dataset

    @staticmethod
    def split(features: np.ndarray, labels: np.ndarray, test_size: float=0.2, random_state=None) -> tuple:
        """
        Splits the dataset into training and testing sets.

        Parameters
        ----------
        features : np.ndarray
            Features of the dataset
        labels : np.ndarray
            Labels of the dataset
        test_size : float
            Size of the testing set

        Returns
        -------
        tuple
            A tuple containing the training and testing sets
        """

        from sklearn.model_selection import train_test_split

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        return features_train, features_test, labels_train, labels_test

    def _detect_csv_delimiter(self) -> str or None:
        delimiters = [",", "\t", ";", "|"]

        try:
            for delimiter in delimiters:
                with open(self.path, "r", encoding="utf-8") as file:
                    first_line = file.readline()

                    for delimiter in delimiters:
                        if delimiter in first_line:
                            return delimiter
        except FileNotFoundError as exc:
            raise FileNotFoundError("The specified file does not exist") from exc

        return None

    def _load(self) -> pd.DataFrame:
        if self.delimiter is None:
            self.delimiter = self._detect_csv_delimiter()

            if self.delimiter is None:
                raise ValueError("Could not detect the delimiter of the CSV file. Please specify it manually")
        try:
            dataset = pd.read_csv(self.path, delimiter=self.delimiter)
            dataset = dataset.dropna()
            dataset = dataset.reset_index(drop=True)

            return dataset
        except FileNotFoundError as exc:
            raise FileNotFoundError("The specified file does not exist") from exc
        except KeyError as exc:
            raise KeyError("The specified target does not exist") from exc

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataset as a Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The dataset
        """

        return self.dataset

    def to_numpy(self, target: str, dataframe: pd.DataFrame=None) -> tuple:
        """
        Converts the dataset to a NumPy array.

        Parameters
        ----------
        target : str
            Name of the target column
        dataframe : pd.DataFrame
            Optional DataFrame to convert

        Returns
        -------
        tuple
            A tuple containing the features and labels
        """

        if dataframe is not None:
            self.dataset = dataframe

        categorical_columns = self.dataset.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            codes, _ = pd.factorize(self.dataset[column])
            self.dataset[column] = codes

        features = self.dataset.drop(target, axis=1).values
        labels = self.dataset[target].values.reshape(-1, 1)

        normalized_features = self.normalize(features)

        return normalized_features, labels
