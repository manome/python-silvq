# -*- encoding: utf8 -*-

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

from lvq import SilvqModel
from lvq.utils import choice_prototypes

def main():
    # Load dataset
    wine = load_wine()
    x = wine.data
    y = wine.target
    # Feature Scaling
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8, shuffle=True, stratify=y)

    # Generating model
    initial_prototypes = choice_prototypes(x_train, y_train, prototypes_per_class=1, random_state=None)
    model = SilvqModel(x.shape[1], theta=0.5, bias_type='ls', initial_prototypes=initial_prototypes)
    # Training the model
    model.fit(x_train, y_train, epochs=30)
    # Predict the response for test dataset
    y_predict = model.predict(x_test)

    # Evaluating the model
    print('Accuracy: %.3f' %accuracy_score(y_test, y_predict))

if __name__ == '__main__':
    main()
