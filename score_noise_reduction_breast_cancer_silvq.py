# -*- encoding: utf8 -*-

import os
import pickle
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from lvq import SilvqModel

def main():
    # Load dataset
    breast_cancer = load_breast_cancer()
    x = breast_cancer.data
    y = breast_cancer.target
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8, shuffle=True, stratify=y)

    # Generating model
    model = SilvqModel(x.shape[1], theta=0.5, bias_type='ls')
    # Training the model
    model.fit(x_train, y_train, epochs=30)
    # Predict the response for test dataset
    y_predict = model.predict(x_test)

    # Evaluating the model
    print('** Original ****************************')
    print('Accuracy: %.3f' %accuracy_score(y_test, y_predict))
    print('Number of prototypes: {}'.format(model.n_prototypes))

    # Noise reduction
    model.delete_prototype(140)
    # Predict the response for test dataset
    y_predict = model.predict(x_test)

    # Evaluating the model
    print('** Noise reduction *********************')
    print('Accuracy: %.3f' %accuracy_score(y_test, y_predict))
    print('Number of prototypes: {}'.format(model.n_prototypes))

    '''
    # Make directory
    os.makedirs('output/', exist_ok=True)

    # Save model
    with open('output/model.pickle', 'wb') as f:
        pickle.dump(model , f)

    # Load model
    with open('output/model.pickle', mode='rb') as f:
        model = pickle.load(f)
    '''

if __name__ == '__main__':
    main()
