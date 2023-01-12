# -*- encoding: utf8 -*-

import numpy as np

from lvq import SilvqModel

def main():
    # Load dataset
    dataset = np.loadtxt('data/artificial_dataset2.csv', delimiter=',')
    x = dataset[:, :-1].astype('float64')
    y = dataset[:, -1].astype('int64')

    # Generating model
    model = SilvqModel(x.shape[1], theta=0.5, bias_type='ls')
    # Training the model
    model.fit(x, y, epochs=30)

    # Noise reduction
    model.delete_prototype(60)

    # Export model as compression & noise reduction data
    model.export_as_compressed_data(path='output/', filename='compressed_and_noise_reduced_data.csv')

if __name__ == '__main__':
    main()
