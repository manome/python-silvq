# -*- encoding: utf8 -*-

# Author: Nobuhito Manome <manome@g.ecc.u-tokyo.ac.jp>
# License: BSD 3 clause

import warnings
import numpy as np
from matplotlib import pyplot as plt

def choice_prototypes(x_train, y_train, prototypes_per_class=1, random_state=None):
    '''
    Choose prototypes randomly from input data.
    ----------
    Parameters
    ----------
    x_train : array-like, shape = [n_samples, n_features]
        Input data.
    y_train : array, shape = [n_samples]
        Input data target.
    prototypes_per_class : int, optional (default=1)
        Number of prototypes per class.
    random_state : int, optional (default=None)
        Random state to choice prototypes.
    ----------
    Returns
    ----------
    prototypes : array-like, shape = [n_prototypes, n_features + 1]
        Returns prototypes for lvq model.
    '''
    # Get unique class labels and their counts in the training data
    y_unique, y_counts = np.unique(y_train, return_counts=True)
    # Determine the minimum number of samples across all classes
    min_y_counts = np.min(y_counts)
    # Randomly select indices for prototypes from the available samples in each class
    idx_rand = np.random.RandomState(random_state).choice(min_y_counts, prototypes_per_class, replace=False)
    # Initialize an empty array to store the prototypes
    prototypes = np.zeros((0, x_train.shape[1] + 1))
    # Loop over the randomly selected indices
    for idx in idx_rand:
        # Loop over each unique class label
        for label in y_unique:
            # Get the index of the sample to use as a prototype for the current class
            idx_label = np.where(y_train == label)[0][idx]
            # Append the sample features and its label to the prototypes array
            prototypes = np.append(prototypes, np.array([np.append(x_train[idx_label], y_train[idx_label])]), axis=0)
    return prototypes

def plot2d(model, x, y, title='dataset'):
    '''
    Projects the input data to two dimensions and plots it.
    The projection is done using the relevances of the given lvq model.
    The plot shows the target class of each data point (big circle) and which class was predicted (smaller circle).
    It also shows the prototypes (diamond).
    ----------
    Parameters
    ----------
    model : lvq model
    x : array-like, shape = [n_samples, n_features]
        Input data.
    y : array, shape = [n_samples]
        Input data target.
    title : str, optional (default='dataset')
        The title to use.
    '''
    y_unique = np.unique(y)
    n_label = y_unique.shape[0]
    cmap = plt.get_cmap('jet')
    # Precompute colors for each label
    colors = [cmap(i / (n_label - 1)) for i in range(n_label)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Plot dataset
    for idx, label in enumerate(y_unique):
        indices = (y == label)
        ax1.plot(x[indices, 0], x[indices, 1], 'o', color=colors[idx], label=str(label))
    ax1.grid()
    ax1.legend(loc='upper left')
    ax1.set_title(title)
    ax1.set_xlabel('1st')
    ax1.set_ylabel('2nd')
    # Predict the response for x
    y_predict = model.predict(x)
    # Plot prediction results
    for idx, label in enumerate(y_unique):
        indices = (y_predict == label)
        ax2.plot(x[indices, 0], x[indices, 1], '.', color='white', markeredgecolor=colors[idx])
    # Plot prototypes
    for idx, label in enumerate(y_unique):
        indices = (model.c == label)
        ax2.plot(model.m[indices, 0], model.m[indices, 1], 'D', color=colors[idx])
    ax2.grid()
    ax2.set_title('Prediction results & Prototypes')
    ax2.set_xlabel('1st')
    ax2.set_ylabel('2nd')
    fig.align_labels([ax1, ax2])
    # Show plot
    plt.show()

def conformal_predict(model, x_calib, y_calib, x_test, confidence_level=0.95, proba_threshold=None, top_k=None, sort_by_proba=True):
    '''
    Generate conformal predictions for each input sample based on the given confidence level.
    ----------
    Parameters
    ----------
    model : object
        The trained model used to make conformal predictions.
    x_calib : array-like, shape = [n_samples, n_features]
        Calibration data.
    y_calib : array, shape = [n_samples]
        True labels for the calibration data.
    x_test : array-like, shape = [n_samples, n_features]
        Test input data for which predictions are to be made.
    confidence_level : float, optional (default=0.95)
        Confidence level for the conformal prediction. It should be between 0 and 1.
    proba_threshold : float, optional (default=None)
        Minimum probability threshold for including a label in the predictions.
    top_k : int, optional (default=None)
        Maximum number of labels to output per test sample.
    sort_by_proba : bool, optional (default=True)
        Whether to sort the output labels by their prediction probabilities.
    ----------
    Returns
    ----------
    conformal_predictions : list of lists
        For each test sample, a list of class indices that meet the conformal prediction criteria.
    '''
    # Calculate the number of samples in the calibration dataset
    n = x_calib.shape[0]
    # Obtain the predicted probabilities for the calibration data
    y_calib_proba = model.predict_proba(x_calib)
    # Calculate the probability of the true class for each calibration sample
    prob_true_class = y_calib_proba[np.arange(n), y_calib]
    # Convert to uncertainty scores by subtracting the true class probabilities from 1
    scores = 1 - prob_true_class
    # Set alpha as the complement of the confidence level
    alpha = 1 - confidence_level
    # Calculate the quantile level for the uncertainty scores
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if not (0 <= q_level <= 1):
        # Apply clipping to ensure q_level is within the range [0, 1]
        warnings.warn(f"Warning: q_level ({q_level}) is out of the range [0, 1]. It will be clipped.")
        q_level = min(max(q_level, 0), 1)
    qhat = np.quantile(scores, q_level, method='higher')
    # Obtain the predicted probabilities for the test data
    y_test_proba = model.predict_proba(x_test)
    # Calculate the uncertainty scores for the test data
    test_scores = 1 - y_test_proba
    # Determine which classes are valid based on the quantile threshold
    valid_classes_matrix = test_scores <= qhat
    # Filter predictions based on the probability threshold, if specified
    if proba_threshold is not None:
        valid_classes_matrix &= (y_test_proba >= proba_threshold)
    # If top_k is specified, limit the number of labels to the top k probabilities
    if top_k is not None:
        top_k_indices = np.argsort(y_test_proba, axis=1)[:, -top_k:]
        top_k_mask = np.zeros_like(y_test_proba, dtype=bool)
        for i, indices in enumerate(top_k_indices):
            top_k_mask[i, indices] = True
        valid_classes_matrix &= top_k_mask
    # Collect the indices of valid classes for each test sample
    conformal_predictions = []
    for i, valid_classes in enumerate(valid_classes_matrix):
        valid_classes_indices = np.where(valid_classes)[0]
        if sort_by_proba:
            # Sort valid classes by their predicted probability
            sorted_classes = sorted(valid_classes_indices, key=lambda x: y_test_proba[i, x], reverse=True)
            conformal_predictions.append(sorted_classes)
        else:
            conformal_predictions.append(valid_classes_indices.tolist())
    return conformal_predictions

def accuracy_score_conformal_predictions(y_test, conformal_predictions):
    '''
    Calculate the accuracy of conformal predictions.
    ----------
    Parameters
    ----------
    y_test : array-like, shape = [n_samples]
        True labels for the test data.
    conformal_predictions : list of lists, length = [n_samples]
        Each entry contains a list of predicted classes for the corresponding sample.
    ----------
    Returns
    ----------
    accuracy : float
        The accuracy of the conformal predictions, i.e., the proportion of test samples
        for which the true label is among the predicted classes.
    '''
    # Convert y_test and conformal_predictions to numpy arrays for easier processing
    y_test = np.asarray(y_test)
    conformal_predictions = np.asarray([set(preds) for preds in conformal_predictions])
    # Check if each true label is among the predicted classes for the corresponding sample
    correct_preds = np.array([y in preds for y, preds in zip(y_test, conformal_predictions)])
    # Calculate the proportion of correct predictions
    accuracy = np.mean(correct_preds)
    return accuracy
