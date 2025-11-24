#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib import interactive

import json
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix
from .OrdinaryLeastSquaresRegression import OrdinaryLeastSquaresRegression
def generate(problem, n_samples, n_features):
    """
    Generate a dataset
    """
    if problem == "classification":
        print("Generate a classification dataset with {} samples, {} features and 2 classes".format(n_samples, n_features))
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2)

    elif problem == "regression" or problem == "ordinaryleastsquaresregression":
        print("Generate a regression dataset with {} samples and {} features".format(n_samples, n_features))
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10)

    else:
        raise Exception("Unknown problem")

    return X, y

def get_metric(problem):
    """
    Get the metric for computing the model performances
    """
    if problem == "classification":
        return accuracy_score

    elif problem == "regression" or problem == "ordinaryleastsquaresregression":
        return mean_squared_error

    else:
        raise Exception("Unknown problem")

def learn(problem, X, y):
    """
    Learn a linear model
    """
    if problem == "classification":
        print("Learning a Linear Discriminant Analysis.", end=" ")
        model = LinearDiscriminantAnalysis()

    elif problem == "regression":
        print("Learning a Linear Regression.", end=" ")
        model = LinearRegression()
    elif problem == "ordinaryleastsquaresregression":
        print("Learning an Ordinary Linear Regression.", end=" ")
        model = OrdinaryLeastSquaresRegression()
    else:
        raise Exception("Unknown problem")

    # Train/test splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    # Model fitting
    model.fit(X_train, y_train)

    # Compute the model error
    y_pred = model.predict(X_test)
    error = get_metric(problem)(y_test, y_pred)
    print("Model error: {:.4f}".format(error))

    if problem == "classification":
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    return model, error

def predict(model, problem):
    """
    Predict new date
    """
    # Generate a new dataset
    print("Predict on a new dataset of 10 values")
    try:
        coefs = model.coef_.ravel()
    except AttributeError:
        coefs = model.coefs_.ravel()
    X, _ = generate(problem, 10, len(coefs))

    # Call the model
    return model.predict(X)

def target_statistics(y):
    """
    Stats for the target vector
    """
    mean_target = np.mean(y)
    std_target = np.std(y)
    print("Statistics of the target. Mean = {}, std = {}".format(mean_target, std_target))
    return mean_target, std_target

def features_statistics(X):
    """
    Stats for the features
    """
    mean_features = {"feature_{}".format(k): np.mean(X[:,k]) for k in range(X.shape[1])}
    print("Mean values of the features:")
    print(json.dumps(mean_features, indent=4))

    std_features = {"feature_{}".format(k): np.std(X[:,k]) for k in range(X.shape[1])}
    print("Std values of the features:")
    print(json.dumps(std_features, indent=4))

    return mean_features, std_features

def correlation(X, y):
    """
    Pearson correlation coefficient
    """
    corr_coefs = {"feature_{}".format(k): np.corrcoef(X[:,k], y) for k in range(X.shape[1])}
    print_correlations(X, y)
    return corr_coefs

def statistics(X, y):
    """
    Compute descriptive statistics from the data
    """
    print("Compute the statistics")
    mean_target, std_target = target_statistics(y)
    mean_features, std_features = features_statistics(X)
    corr_coefs = correlation(X, y)

    return {
        "mean_target": mean_target,
        "std_target": std_target,
        "mean_features": mean_features,
        "std_features": std_features,
        "correlations": corr_coefs
    }

def print_correlations(X, y):
    """
    Print the correlation of the dataset
    """
    dataset = np.concatenate([X, y.reshape(-1,1)], axis=1)
    print("Dataset correlation matrix:")
    print(np.round(np.corrcoef(dataset), 2))

if __name__ == "__main__":

    n_samples = 10
    n_features = 5
    interactive(True)

    for problem in ["classification", "regression","ordinaryleastsquaresregression"]:
        X, y = generate(problem, n_samples=n_samples, n_features=n_features)
        stats = statistics(X, y)
        model, error = learn(problem, X, y)
        predictions = predict(model, problem)
