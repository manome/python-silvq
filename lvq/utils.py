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
    y_unique, y_counts = np.unique(y_train, return_counts=True)
    min_y_counts = np.min(y_counts)
    idx_rand = np.random.RandomState(random_state).choice(min_y_counts, prototypes_per_class, replace=False)
    prototypes = np.zeros((0, x_train.shape[1] + 1))
    for idx in idx_rand:
        for label in y_unique:
            idx_label = np.where(y_train == label)[0][idx]
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
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Plot dataset
    for idx, label in enumerate(y_unique):
        indices = np.where(y == label)[0]
        color = cmap(idx/(n_label - 1))
        ax1.plot(x[indices, 0], x[indices, 1], 'bo', color=color, label=str(label), marker='o')
    ax1.grid()
    ax1.legend(loc='upper left')
    ax1.set_title(title)
    ax1.set_xlabel('1st')
    ax1.set_ylabel('2nd')
    # Predict the response for x
    y_predict = model.predict(x)
    # Plot prediction results
    for idx, label in enumerate(y_unique):
        indices = np.where(y_predict == label)[0]
        color = cmap(idx/(n_label - 1))
        ax2.plot(x[indices, 0], x[indices, 1], 'bo', color='white', marker='.', markeredgecolor=color)
    # Plot prototypes
    for idx, label in enumerate(y_unique):
        indices = np.where(model.c == label)[0]
        color = cmap(idx/(n_label - 1))
        ax2.plot(model.m[indices, 0], model.m[indices, 1], 'bo', color=color, marker='D')
    ax2.grid()
    ax2.set_title('Prediction results & Prototypes')
    ax2.set_xlabel('1st')
    ax2.set_ylabel('2nd')
    fig.align_labels([ax1, ax2])
    # Show plot
    plt.show()
