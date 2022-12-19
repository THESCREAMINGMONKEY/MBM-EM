from itertools import islice
from random import random

import numpy as np
from numpy import size, arange
from scipy import sparse

from sklearn.mixture._base import BaseMixture
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import BernoulliNB, _BaseDiscreteNB
from sklearn.utils import _print_elapsed_time
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES, check_X_y
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

seed = 42
np.random.seed(seed)
rand = random()


# Defining a class for the EM algorithm.
class MBM_EM(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True
        '''self.X = None
        self.N = None
        self.D = None
        self.priors = []
        self.p = None
        self.ll = None
        self.classes_ = []
        self.n_classes_ = None'''

    def fit(self, X, y):

        X, y = check_X_y(X, y)

        unique,counts = np.unique(y,return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        it = 0

        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.ll = []
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)

        # Inizialize params
        self.priors = [c0 / (c1 + c0), c1 / (c1 + c0)]
        self.p = np.random.uniform(size=(self.n_classes_, self.D))

        '''
        print("MBMEM PRIORS\n", self.priors)
        print("MBMEM P SHAPE\n", self.p.shape)
        print("MBMEM P SIZE\n", self.p.size)
        print("MBMEM P\n", self.p)
        print("MBMEM X SHAPE\n", self.X.shape)
        '''

        # Training loop.
        for i in range(self.N):

            # E-step

            # P(X} | {y})
            # OLD
            prob = logsumexp((np.dot(self.X, np.log(self.p.T)) + (np.dot((1 - self.X), np.log((1 - self.p.T)))))) #700x2

            '''
            print("BERNOULLI D(X|y)\n", berD)
            print("BERNOULLI D(X|y) SHAPE\n", berD.shape)
            '''


            # Calculating the resp P(Z | X, p_, priors).
            num_resp = prob * self.priors # 700x2
            den_resp = np.sum(num_resp, axis=1, keepdims=True) #700x1
            resp = num_resp / den_resp #700x2


            '''
            print("den_resp shape\n", den_resp.shape)
            print("num_resp shape\n", num_resp.shape)
            print("RESP SHAPE\n", resp.shape)
            
            print("RESP\n", resp)
            '''


            # Calc ll
            self.ll.append(np.sum(np.log(den_resp)))

            # Stop condition
            if len(self.ll) > 1:
                delta = self.ll[-1] - self.ll[-2]
                it = it + 1

                if delta <= self.min_change:
                #if it == self.max_it:
                    return self

            # Printing the results
            if self.verbose:
                print(f'Iteration: {i :4d} | Log likelihood: {self.ll[i] : 07.5f}')

            # M-step

            # priors_(t)
            sum_resp = np.sum(resp, axis=0, keepdims=True) #1x2
            self.priors = sum_resp / self.N #1x2

            #print("MBMEM NEW PRIORS\n", self.priors)

            # p_(t)
            self.p = np.dot(resp.T, self.X) / sum_resp.T #2x13 nan

            #print("MBMEM NEW P\n", self.p)

        return self


    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = []

        #print("LEN X\n", len(X))
        #print("SIZE X\n", size(X, 0))

        for i in range(size(X, 0)):
            probas = []
            for j in range(self.n_classes_):

                '''
                print("PREDICT PRIORS j\n", self.priors.T[j])
                print("PREDICT PRIORS SHAPE\n", self.priors.shape)
                print("PREDICT P[j]\n", self.p[j])
                print("PREDICT X[i]\n", X[i])
                '''

                probas.append((self.p[j]*X[i] + (1-self.p[j]) * (1-X[i])).prod() *
                              self.priors.T[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())


        return np.array(res)


    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)

