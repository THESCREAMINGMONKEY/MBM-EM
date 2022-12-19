#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:46:43 2022

@author: nico
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#from sklearn.utils.multiclass import unique_labels


# generate random floating point values
import random
from random import seed


class BNB(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique,counts = np.unique(y,return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        cn = counts[unique.index(-1)]
        self.classes_ = [-1,0,1]
        self.n_classes_ = len(self.classes_)

        '''print('SELF N CLASSES')
        print(self.n_classes_)

        print('UNIQUE INDEX 0')
        print(unique.index(0))
        print('UNIQUE INDEX 1')
        print(unique.index(1))
        print('UNIQUE INDEX -1')
        print(unique.index(-1))'''


        random.seed(1)
        rand_priors = random.random()
        rand_p = random.random()

        self.priors_ = [c0 / (c1 + c0), c1 / (c1 + c0)]
        self.p_ = np.array([X for i in range(self.n_classes_)])


        print("__[p, priors y != -1] __")
        print(self.p_)
        print(self.p_.shape)
        print(self.priors_)
        print("__________ FINE ________")



        for i in range(self.n_classes_):
            if y[i] == -1:

                self.priors_ = [1 - rand_priors, rand_priors]
                self.p_ = np.array([X[np.where(y==-1)] for i in range(self.n_classes_)])
                print("__[p, priors y == -1] __")
                print(self.p_)
                print('SHAPE' + np.size(self.p_))
                print(self.priors_)
                print("__________ FINE ________")
                for j in range(30):

                    #E
                    for i in range(self.n_classes_):
                        self.resp_ = self.priors_[1] * (((self.p_)**(X[np.where(y==i)]) * (1 - self.p_)**(1 - X[np.where(y==i)])).prod())

                    #M
                    for i in range(self.n_classes_):
                        self.priors_ = [self.resp_.mean()]
                        self.p_ = np.array([X[np.where(y==i).mean(axis=0)] * self.resp_])

        return self



    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = []
        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):
                probas.append((self.p_[j]*X[i] +
                               (1-self.p_[j])*(1-X[i])).prod()
                              *self.priors_[j])
            probas = np.array(probas)
            res.append(probas / probas.sum())


        return np.array(res)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)