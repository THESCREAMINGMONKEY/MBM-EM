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


class BNB(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique,counts = np.unique(y,return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        self.priors_ = [c0 / (c1 + c0), c1 / (c1 + c0)]
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)
        self.p_ = np.array([X[np.where(y==i)].mean(axis=0) for i in range(self.n_classes_)])

        '''
        print("BNB P")
        print(self.pi.shape)
        print(self.pi.size)
        
        print(self.p_.shape)
        print(self.p_.size)
        '''

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = []
        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):

                '''
                print(self.priors_[j])
                print(self.p_[j])
                print(X[i])
                '''

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