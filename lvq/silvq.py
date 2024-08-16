# -*- encoding: utf8 -*-

# Author: Nobuhito Manome <manome@g.ecc.u-tokyo.ac.jp>
# License: BSD 3 clause

import os
import numpy as np

class SilvqModel():
    '''
    Self-incremental learning vector quantization Model (SilvqModel)
    ----------
    Parameters
    ----------
    n_features : int
        Number of features.
    theta : float, value >= 0.5, optional (default=0.5)
        Threshold for adding prototypes.
    bias_type : str, value = 'cp' or 'rs' or 'ls' or 'dp' or 'dfh' or 'paris', optional (default='ls')
        Types of causal induction model.
    initial_prototypes : array-like, shape = [n_prototypes, n_features + 1], optional (default=None)
        Prototypes to start with.
        Class label must be placed as last entry of each prototype.
    max_n_prototypes : int, optional (default=50000)
        Maximum number of prototypes.
    low_memory_mode : bool, optional (default=False)
        If True, enable low memory mode to reduce memory usage.
    '''
    def __init__(self, n_features, theta=0.5, bias_type='ls', initial_prototypes=None, max_n_prototypes=50000, low_memory_mode=False):
        if initial_prototypes is None:
            self.n_prototypes = 0 # Number of prototypes
            self.m = np.zeros((0, n_features)) # Prototype vector
            self.c = np.zeros(0) # Class label
        else:
            self.n_prototypes = initial_prototypes.shape[0]
            self.m = initial_prototypes[:, :-1]
            self.c = initial_prototypes[:, -1]
        self.cooccur_a = np.zeros(self.n_prototypes) # Co-occurrence frequency information for each prototype
        self.cooccur_b = np.zeros(self.n_prototypes)
        self.cooccur_c = np.zeros(self.n_prototypes)
        self.cooccur_d = np.zeros(self.n_prototypes)
        self.r = np.zeros(self.n_prototypes) # Label confidence (strength of the causal relationship)
        self.alpha = np.zeros(self.n_prototypes) # Learning rate
        self.t = np.zeros(self.n_prototypes) # Number of learning times
        self.theta = theta # Threshold for adding prototypes
        self.bias_type = bias_type # Types of causal induction model
        self.max_n_prototypes = max_n_prototypes # Maximum number of prototypes
        self.low_memory_mode = low_memory_mode # Whether to use low memory mode

    def export_as_compressed_data(self, path='output/', filename='compressed_data.csv'):
        '''
        export model as compressed data.
        ----------
        Parameters
        ----------
        path : str, optional (default='output/')
            Save path.
        filename : str, optional (default='compressed_data.csv')
            Save filename.
        '''
        os.makedirs(path, exist_ok=True)
        data = np.hstack([self.m, self.c.reshape(-1, 1)])
        np.savetxt(os.path.join(path, filename), data, delimiter=',')
        print(f'export model as compressed data. (file: {path}{filename})')

    def delete_prototype(self, age):
        '''
        delete prototypes below a certain number of learning times.
        ----------
        Parameters
        ----------
        age : int
            Number of learning times.
        '''
        idx = np.where(self.t <= age)
        self.cooccur_a = np.delete(self.cooccur_a, idx[0], axis = 0)
        self.cooccur_b = np.delete(self.cooccur_b, idx[0], axis = 0)
        self.cooccur_c = np.delete(self.cooccur_c, idx[0], axis = 0)
        self.cooccur_d = np.delete(self.cooccur_d, idx[0], axis = 0)
        self.r = np.delete(self.r, idx[0], axis = 0)
        self.alpha = np.delete(self.alpha, idx[0], axis = 0)
        self.m = np.delete(self.m, idx[0], axis = 0)
        self.c = np.delete(self.c, idx[0], axis = 0)
        self.t = np.delete(self.t, idx[0], axis = 0)
        self.n_prototypes -= idx[0].shape[0]

    def add_prototype(self, x, c):
        if self.n_prototypes < self.max_n_prototypes:
            if np.any(self.c == c):
                self.cooccur_a = np.append(self.cooccur_a, self.cooccur_a[np.where(self.c == c)[0][0]])
                self.cooccur_b = np.append(self.cooccur_b, self.cooccur_b[np.where(self.c == c)[0][0]])
                self.cooccur_c = np.append(self.cooccur_c, self.cooccur_c[np.where(self.c == c)[0][0]])
                self.cooccur_d = np.append(self.cooccur_d, self.cooccur_d[np.where(self.c == c)[0][0]])
                self.r = np.append(self.r, self.r[np.where(self.c == c)[0][0]])
                self.alpha = np.append(self.alpha, self.alpha[np.where(self.c == c)[0][0]])
            else:
                self.cooccur_a = np.append(self.cooccur_a, 0)
                self.cooccur_b = np.append(self.cooccur_b, 0)
                self.cooccur_c = np.append(self.cooccur_c, 0)
                self.cooccur_d = np.append(self.cooccur_d, 0)
                self.r = np.append(self.r, 0)
                self.alpha = np.append(self.alpha, 0)
            self.m = np.append(self.m, np.array([x]), axis=0)
            self.c = np.append(self.c, c)
            self.t = np.append(self.t, 0)
            self.n_prototypes += 1

    def update_alpha(self):
        self.alpha = 1.0 - self.r

    def update_r(self):
        with np.errstate(all='ignore'):
            if self.bias_type == 'cp':
                self.r = self.cooccur_a / (self.cooccur_a + self.cooccur_b)
            elif self.bias_type == 'rs':
                self.r = (self.cooccur_a + self.cooccur_d) / (self.cooccur_a + self.cooccur_b + self.cooccur_c + self.cooccur_d)
            elif self.bias_type == 'ls':
                self.r = (self.cooccur_a + (self.cooccur_b / (self.cooccur_b + self.cooccur_d)) * self.cooccur_d) / (self.cooccur_a + self.cooccur_b + (self.cooccur_a / (self.cooccur_a + self.cooccur_c)) * self.cooccur_c + (self.cooccur_b / (self.cooccur_b + self.cooccur_d)) * self.cooccur_d)
            elif self.bias_type == 'dp':
                self.r = (((self.cooccur_a * self.cooccur_d - self.cooccur_b * self.cooccur_c)/((self.cooccur_a + self.cooccur_b) * (self.cooccur_c + self.cooccur_d))) + 1) / 2
            elif self.bias_type == 'dfh':
                self.r = self.cooccur_a / np.sqrt((self.cooccur_a + self.cooccur_b) * (self.cooccur_a + self.cooccur_c))
            elif self.bias_type == 'paris':
                self.r = self.cooccur_a / (self.cooccur_a + self.cooccur_b + self.cooccur_c)
            self.r[np.isnan(self.r)] = 0

    def update_cooccur(self, idx_c_win, c_win, c):
        if c_win == c:
            self.cooccur_a[np.where(self.c == c_win)] += 1
            self.cooccur_d[np.where(self.c != c_win)] += 1
        else:
            self.cooccur_b[np.where(self.c == c_win)] += 1
            self.cooccur_c[np.where(self.c == c)] += 1
            self.cooccur_d[np.where(self.c != c_win)] += 1
            self.cooccur_d[np.where(self.c == c)] -= 1

    def learn_one(self, x, c):
        if np.any(self.c == c):
            dist = np.linalg.norm(x - self.m, axis=1)
            idx_c_win = np.argsort(dist)[0]
            c_win = self.c[idx_c_win]
            idx_c = np.where(self.c == c)[0]
            idx_dist_min_c = idx_c[np.argsort(dist[idx_c])[0]]
            self.t[idx_c_win] += 1
            if c_win != c and self.r[idx_c_win] > self.theta:
                self.add_prototype(x, c)
            else:
                self.update_cooccur(idx_c_win, c_win, c)
                self.update_r()
                self.update_alpha()
                self.m[idx_dist_min_c] = self.m[idx_dist_min_c] + self.alpha[idx_dist_min_c] * (x - self.m[idx_dist_min_c])
        else:
            self.add_prototype(x, c)

    def fit(self, x_train, y_train, epochs=30):
        '''
        Fit the model to the given training data.
        ----------
        Parameters
        ----------
        x_train : array-like, shape = [n_samples, n_features]
            Input data.
        y_train : array, shape = [n_samples]
            Input data target.
        epochs : int, optional (default=30)
            Number of epochs.
        '''
        for i in range(epochs):
            for j in range(x_train.shape[0]):
                self.learn_one(x_train[j], y_train[j])

    def predict_one(self, x):
        '''
        Predict the class label for a single input sample.
        ----------
        Parameters
        ----------
        x : array-like, shape = [n_features]
            Single input data sample.
        ----------
        Returns
        ----------
        class_label : The predicted class label for the input sample.
        '''
        # Calculate the Euclidean distance between the input sample x and each vector in self.m
        dist = np.linalg.norm(x - self.m, axis=1)
        # Find the index of the vector in self.m that has the minimum distance to the input sample
        idx_c_win = np.argmin(dist)
        # Return the class label corresponding to the closest vector
        return self.c[idx_c_win]

    def predict(self, x_test):
        '''
        Predict class label for each input sample.
        ----------
        Parameters
        ----------
        x_test : array-like, shape = [n_samples, n_features]
            Input data.
        ----------
        Returns
        ----------
        y_predict : array, shape = [n_samples]
            Returns predicted values.
        '''
        if self.low_memory_mode:
            # Predict the response for test dataset
            y_predict = np.zeros(x_test.shape[0])
            for i, x in enumerate(x_test):
                y_predict[i] = self.predict_one(x)
        else:
            # Calculate the distance from each sample in x_test to each vector in self.m
            dist = np.linalg.norm(x_test[:, np.newaxis] - self.m, axis=2)
            # Find the index of the minimum distance for each sample
            idx_c_win = np.argmin(dist, axis=1)
            # Use the indices to get the corresponding class predictions
            y_predict = self.c[idx_c_win]
        return y_predict

    def predict_proba_one(self, x):
        '''
        Predict class probabilities for a single input sample.
        ----------
        Parameters
        ----------
        x : array-like, shape = [n_features]
            Single input data sample.
        ----------
        Returns
        ----------
        probabilities : array, shape = [n_classes]
            Returns the predicted class probabilities for the input sample.
        '''
        # Calculate the Euclidean distance between the input sample x and each vector in self.m
        dist = np.linalg.norm(x - self.m, axis=1)
        # For each unique class, find the minimum distance from the input sample to any vector in that class
        dist_sums = np.array([np.min(dist[self.c == c]) for c in np.unique(self.c)])
        # Replace zeros in dist_sums with a small value to avoid division by zero
        dist_sums[dist_sums == 0] = 1e-10
        # Calculate the probabilities by taking the inverse of these minimum distances
        probabilities = 1 / dist_sums
        # Normalize the probabilities so that they sum to 1
        probabilities /= np.sum(probabilities)
        return probabilities

    def predict_proba(self, x_test):
        '''
        Predict class probabilities for each input sample.
        ----------
        Parameters
        ----------
        x_test : array-like, shape = [n_samples, n_features]
            Input data.
        ----------
        Returns
        ----------
        y_predict_proba : array, shape = [n_samples, n_classes]
            Returns class probabilities for each input sample.
        '''
        if self.low_memory_mode:
            # Predict the response for test dataset
            y_predict_proba = np.zeros([x_test.shape[0], np.unique(self.c).shape[0]])
            for i, x in enumerate(x_test):
                y_predict_proba[i] = self.predict_proba_one(x)
        else:
            # Calculate the distance from each sample in x_test to each vector in self.m
            dist = np.linalg.norm(x_test[:, np.newaxis] - self.m, axis=2)
            # Get unique classes and create a mask for each class
            unique_classes = np.unique(self.c)
            class_masks = {c: self.c == c for c in unique_classes}
            # Initialize an array to store the minimum distances for each class
            dist_sums = np.zeros((x_test.shape[0], unique_classes.shape[0]))
            # For each class, find the minimum distance for each sample
            for idx, c in enumerate(unique_classes):
                dist_sums[:, idx] = np.min(dist[:, class_masks[c]], axis=1)
            # Replace zeros in dist_sums with a small value to avoid division by zero
            dist_sums[dist_sums == 0] = 1e-10
            # Calculate probabilities by taking the inverse of distances
            y_predict_proba = 1 / dist_sums
            # Normalize the probabilities
            y_predict_proba /= np.sum(y_predict_proba, axis=1, keepdims=True)
        return y_predict_proba
