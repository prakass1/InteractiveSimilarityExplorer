'''
Acknowledgement: I would like to thank the author for the idea and the original paper.
Reference Python implementation: https://github.com/KacperKubara/distython/blob/master/HEOM.py
'''

import numpy as np


class HEOM:
    def __init__(self, X, cat_idx, num_idx):
        '''
        :param X: The dataframe to be used. Note: It must be encoded or it wont be beneficial.
        :param cat_idx: The categorical column indexes
        :param num_idx: The numerical column indexes
        '''
        self.cat_idx = cat_idx
        self.num_idx = num_idx
        self.range = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        self.n = X.shape[0]
        self.d = X.shape[1]

    def heom_distance(self, x, y=None):
        '''
        :param x: Array containing the questionnaires
        :param y: Array containing the questionnaires. But, this is optional and not necessary.
        :return:
        '''
        #Results to be appended upon for final calculation
        results = np.zeros(x.shape)
        # Case-1: To find the distance between two categorical attributes.
        # A simple overlap metric weighted by attribute dimensions.
        results[self.cat_idx] = (np.not_equal(x[self.cat_idx], y[self.cat_idx]) * 1) / self.d
                                #/ self.d
        # Case-2: For numerical attributes take the difference and divide by range(max - min) normalization.
        if len(self.num_idx) > 0:
            results[self.num_idx] = np.divide(np.abs(x[self.num_idx] - y[self.num_idx]), self.range[self.num_idx])

        # Final calculate the sqrt(square(results)).
        # We attempt to return the sqrt as per original implementation.
        return np.sqrt(np.sum(np.square(results)))
