# -*- encoding: utf8 -*-

# Author: Nobuhito Manome <manome@g.ecc.u-tokyo.ac.jp>
# License: BSD 3 clause

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
    y_test = np.asarray(y_test)
    conformal_predictions = np.asarray([set(preds) for preds in conformal_predictions])
    correct_preds = np.array([y in preds for y, preds in zip(y_test, conformal_predictions)])
    accuracy = np.mean(correct_preds)
    return accuracy
