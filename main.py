from sklearn.datasets import make_blobs
from multiclass_logistic_regression import MulticlassLogisticRegression

if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    model = MulticlassLogisticRegression()
    model.train(X, y)
    model.display_dataset(X, y)
    model.display_learning_curves()
