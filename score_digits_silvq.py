# -*- encoding: utf8 -*-

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from lvq import SilvqModel

def main():
    # Load dataset
    digits = load_digits()
    x = digits.data
    y = digits.target
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8, shuffle=True, stratify=y)

    # Generating model
    model = SilvqModel(x.shape[1], theta=0.8, bias_type='ls')
    # Training the model
    model.fit(x_train, y_train, epochs=1)
    # Predict the response for test dataset
    y_predict = model.predict(x_test)

    # Evaluating the model
    print('Accuracy: %.3f' %accuracy_score(y_test, y_predict))

if __name__ == '__main__':
    main()
