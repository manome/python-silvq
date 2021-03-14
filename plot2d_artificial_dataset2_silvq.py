# -*- encoding: utf8 -*-

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lvq import SilvqModel
from lvq.utils import plot2d

def main():
    # Load dataset
    dataset = np.loadtxt('data/artificial_dataset2.csv', delimiter=',')
    x = dataset[:, :-1].astype('float64')
    y = dataset[:, -1].astype('int64')
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4, shuffle=True, stratify=y)

    # Generating model
    model = SilvqModel(x.shape[1], theta=0.8, bias_type='ls')
    # Training the model
    model.fit(x_train, y_train, epochs=30)
    # Predict the response for test dataset
    y_predict = model.predict(x_test)

    # Evaluating the model
    print('Accuracy: %.3f' %accuracy_score(y_test, y_predict))
    # Plot prediction results and prototypes
    plot2d(model, x, y, title='Artificial dataset2')

if __name__ == '__main__':
    main()
