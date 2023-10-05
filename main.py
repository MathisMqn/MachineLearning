from sklearn.datasets import make_blobs
from logistic_regression import LogisticRegression

if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    model = LogisticRegression()
    model.train(X, y)
    model.display_learning_curves()
    model.display_roc_curve(X, y)
