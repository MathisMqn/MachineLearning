import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def _detect_csv_delimiter(path):
    delimiters = [",", "\t", ";", "|"]

    with open(path, "r", encoding="utf-8") as file:
        first_line = file.readline()
        for delimiter in delimiters:
            if delimiter in first_line:
                return delimiter

def load_dataset(path, target, delimiter=None):
    if delimiter is None:
        delimiter = _detect_csv_delimiter(path)

        if delimiter is None:
            raise Exception("Could not detect the delimiter of the CSV file. Please specify it manually")
    try:
        dataset = pd.read_csv(path, delimiter=delimiter)
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)

        categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            codes, _ = pd.factorize(dataset[column])
            dataset[column] = codes

            if len(dataset[column].unique()) == len(dataset[column]):
                dataset = dataset.drop(column, axis=1)

        X = dataset.drop(target, axis=1).values
        y = dataset[target].values.reshape(-1, 1)

        return X, y
    except FileNotFoundError:
        raise Exception("The specified file does not exist")
    except KeyError:
        raise Exception("The specified target column does not exist")
    
def normalize_dataset(X):
    normalized_dataset = (X - np.min(X)) / (np.max(X) - np.min(X))
    
    return normalized_dataset
    
def display_dataset(X, y):
    if X.shape[1] == 2:
        unique_classes = np.unique(y)

        plt.figure(figsize=(8, 6))

        for label in unique_classes:
            plt.scatter(X[y.flatten() == label, 0], X[y.flatten() == label, 1], label=f"Class {label}")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Scatter plot of dataset")
        plt.legend(loc="best")
        plt.show()
    else:
        raise Exception("You can display a dataset only if it contains 2 features")
    
def get_confusion_matrix(y, y_pred):
    matrix = confusion_matrix(y, y_pred)

    return matrix

def get_accuracy_score(y, y_pred):
    accuracy = accuracy_score(y, y_pred)

    return accuracy
