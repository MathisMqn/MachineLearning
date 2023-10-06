import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


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
