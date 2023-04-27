#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:46:43 2022

@author: nico
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy.special import logsumexp




class BNB(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique,counts = np.unique(y,return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        self.X_ = X
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)

        '''X[X == 0.5] = np.nan
        imp_val = IterativeImputer(random_state=0)
        #imp_val = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_val.fit(X)
        self.X_ = imp_val.transform(X)'''

        '''self.X_ = X
        self.N_ = self.X_.shape[0]
        self.D_ = self.X_.shape[1]
        self.X_ = self.EM_inc_values()'''

        self.p_ = np.array([self.X_[np.where(y==i)].mean(axis=0) for i in range(self.n_classes_)])
        self.priors_ = [c0 / (c1 + c0), c1 / (c1 + c0)]


        return self

    def EM_inc_values(self):

        """Implementation of the Expectation Maximization algorithm for imputing missing values in binary data.

         Returns:
         imputed_data: The data matrix with imputed missing values (N X D matrix)"""

        num_iterations = 100
        tol = 0.0001

        #num_samples, num_features = self.X_.shape

        self.X_[self.X_ == 0.5] = np.nan

        # Initialize the probability of each feature being 1 using the observed data
        prob = np.nanmean(self.X_, axis=0)

        # Initialize the missing values with the mean of the observed values
        imputed_data = np.copy(self.X_)

        imputed_data[np.isnan(imputed_data)] = np.random.binomial(n=1, p=np.tile(prob, (self.N_, 1)))[np.isnan(imputed_data)]

        # Iterate until convergence or maximum number of iterations is reached
        for i in range(num_iterations):

            # Expectation step: calculate the expected value of each missing value
            expected_value = prob[np.newaxis, :] * imputed_data

            for j in range(self.D_):
                missing_values = np.isnan(self.X_[:, j])

                if np.sum(missing_values) > 0:
                    missing_prob = prob[j] * (1 - prob[j]) ** np.sum(missing_values)
                    expected_value[missing_values, j] = missing_prob * np.sum(imputed_data[~missing_values, j]) / np.sum(1 - np.isnan(self.X_[:, j]))

            # Maximization step: update the probability of each feature being 1
            prob = np.mean(expected_value, axis=0)

            # Update the missing values using the expected value
            imputed_data[np.isnan(self.X_)] = np.random.binomial(n=1, p=expected_value)[np.isnan(self.X_)]

            # Check for convergence

            log_likelihood = np.sum(self.X_ * np.log(expected_value + 1e-9) + (1 - self.X_) * np.log(1 - expected_value + 1e-9))
            #log_likelihood = np.sum(self.X_ * expected_value + (1 - self.X_) * (1 - expected_value)) - self.N_ * logsumexp(0, b=np.array([1, np.exp(expected_value)]))

            if i > 0 and np.all(log_likelihood - prev_log_likelihood < tol):
                break
            prev_log_likelihood = log_likelihood

        return imputed_data

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