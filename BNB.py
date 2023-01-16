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
from tabulate import tabulate


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

        # Not work well, mainly for NTNames
        # self.printParams()

        return self

    def printParams(self):

        table = [['Class', 'Priors', 'feat.1', 'feat.2', 'feat.3', 'feat.4', 'feat.5', 'feat.6', 'feat.7', 'feat.8', 'feat.9', 'feat.10', 'feat.11', 'feat.12', 'feat.13'],

                 ['C=0', self.priors_[0], self.p_[0, 0], self.p_[0, 1], self.p_[0, 2],
                  self.p_[0, 3], self.p_[0, 4], self.p_[0, 5], self.p_[0, 6], self.p_[0, 7],
                  self.p_[0, 8], self.p_[0, 9], self.p_[0, 10], self.p_[0, 11], self.p_[0, 12]],

                 ['C=1', self.priors_[1], self.p_[1, 0], self.p_[1, 1], self.p_[1, 2],
                  self.p_[1, 3], self.p_[1, 4], self.p_[1, 5], self.p_[1, 6], self.p_[1, 7],
                  self.p_[1, 8], self.p_[1, 9], self.p_[1, 10], self.p_[1, 11], self.p_[1, 12]]]

        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    ################## PREDICT ##################

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = []
        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):

                '''probas.append((self.p_[j]*X[i] +
                               (1-self.p_[j])*(1-X[i])).prod()
                              *self.priors_[j])'''

                probas.append((self.p_[j]**X[i] * (1-self.p_[j])**(1-X[i])).prod()*self.priors_[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())

        return np.array(res)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)