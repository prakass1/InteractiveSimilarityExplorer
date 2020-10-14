# Class: RandomNeighbors
# Description: Creation of a baseline to compute random neighbors on a defined train and test data.
# Works in similar fashion as sklearn classes. Extend fit() to anything else for other purposes


import numpy as np
from sklearn.base import BaseEstimator

# Note: Extend base estimator if any fit method is being utilized to return a self object.


class RandomNeighbors:
    def __init__(self, x_train, kneighbors=5):
        '''
        Initialize the random class using the train data and number of neighbors
        :param k_rneighbors:
        '''
        self.kneighbors = kneighbors
        self.X_train = x_train
        self.X_train_index = self.fit(x_train)

    # train data can be fetched from self itself, however one can use just the fit hence, added it separate.
    def fit(self, x_train):
        '''
        A fit class is created in the same fashion as sklearn type.
        Just extend this method to perform anything else to the random init
        :param X:
        :return:
        '''

        # Take the index and convert to list and return as part of the fit
        x_train_index = x_train.index.tolist()
        return x_train_index

    def get_random_neighbors(self, X_test):
        '''
        Use the already created index and
        return random 5 neighbors for each of the test data.
        :return:
        '''

        idx_list = list()

        # Uniform distribution based random neighbors are returned
        # replace false indicates that no duplicates are returned
        idx_list = [self.X_train.loc[np.random.choice(self.X_train_index,
                                                      self.kneighbors,
                                                      replace=False)]
                        .index.tolist()
                    for _ in range(0, len(X_test))]

        return idx_list





