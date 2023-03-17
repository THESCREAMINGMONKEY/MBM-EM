"""
Università degli Studi di Bari Aldo Moro

@author: Christian Riefolo
Mat: 618687
email: c.riefolo2@studenti.uniba.it
"""


import numpy as np

from random import random
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from scipy.spatial import distance
from tabulate import tabulate


class BNB_EM(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True

    # ONLY FOR INITIALIZATION INSIDE "fit" METHOD
    def searchUnlabeled(self, y):

        """This function separate the labeled data from the unlabeled in two arrays"""

        N = self.N_
        labeled_X = []
        unlabeled_X = []

        for n in range(N):
            if y[n] == 0:
                labeled_X.append(self.X_[n])
            if y[n] == 1:
                labeled_X.append(self.X_[n])
            if y[n] == -1:
                unlabeled_X.append(self.X_[n])

        return np.array(labeled_X), np.array(unlabeled_X)

    # ONLY FOR INITIALIZATION INSIDE "fit" METHOD
    def learnParams(self, y):

        """Use the labeled data to initialize params"""

        ### E STEP (INIT - RESP OF ONLY LABELED DATA [1 = C|c]) ###

        N = self.N_
        labeled_resp = []

        for n in range(N):
            if y[n] == 0:
                labeled_resp.append([1,0])
            if y[n] == 1:
                labeled_resp.append([0,1])

        ### M STEP (INIT - LEARNING PARAMS FROM LABELED DATA) ###

        N = self.labeled_X_.shape[0]
        D = self.D_
        C = self.n_classes_
        labeled_resp_array = np.array(labeled_resp)

        # Numerator priors
        N_c = np.sum(labeled_resp_array, axis=0)

        p = np.empty((C, D))

        for c in range(C):
            try:
                # Sum is over N data points
                p[c] = np.sum(labeled_resp_array[:, c][:, np.newaxis] * self.labeled_X_, axis=0) / N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        return labeled_resp_array, N_c/N, p

    def EM_inc_values(self):

        """Implementation of the Expectation Maximization algorithm for imputing missing values in binary data.

         Returns:
         imputed_data: The data matrix with imputed missing values (N X D matrix)"""

        num_iterations = 100
        tol = 0.0001

        num_samples, num_features = self.X_.shape

        self.X_[self.X_ == 0.5] = np.nan

        # Initialize the probability of each feature being 1 using the observed data
        prob = np.nanmean(self.X_, axis=0)

        # Initialize the missing values with the mean of the observed values
        imputed_data = np.copy(self.X_)
        imputed_data[np.isnan(imputed_data)] = np.random.binomial(n=1, p=np.tile(prob, (num_samples, 1)))[np.isnan(imputed_data)]

        # Iterate until convergence or maximum number of iterations is reached
        for i in range(num_iterations):

            # Expectation step: calculate the expected value of each missing value
            expected_value = prob[np.newaxis, :] * imputed_data

            for j in range(num_features):
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
            if i > 0 and log_likelihood - prev_log_likelihood < tol:
                break
            prev_log_likelihood = log_likelihood

        return imputed_data



    def fit(self, X, y):

        """EM algorithm of Multivariate of Bernoulli Distributions"""

        X, y = check_X_y(X, y)
        self.X_ = X
        self.N_ = X.shape[0] # Number of individuals
        self.D_ = X.shape[1] # Number of features
        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_) # Number of classes


        ## E-M FOR UNKNOWN VALUE IN X (x_i) ##
        self.X_ = self.EM_inc_values()

        # Create two arrays, one for unlabeled data and the other for the labeled
        self.labeled_X_, self.unlabeled_X_ = self.searchUnlabeled(y)

        # E STEP AND M STEP INIT ON ONLY LABELED DATA
        self.resp_, self.priors_, self.p_ = self.learnParams(y)

        ### E STEP OF UNLABELED AND LABELED DATA ###
        self.resp_, self.likelihood_ = self.EStep(y)

        ### SET STOP CONDITION PARAMS (INIT) ###
        eucl_norm_old = self.stopCondition()

        '''############################### EM ALGO LOOP ###############################'''

        for i in range(self.max_it):

            ### M STEP ###
            self.priors_, self.p_ = self.MStep()

            ### E STEP ###
            self.resp_, self.likelihood_ = self.EStep(y)

            ### SET STOP CONDITION PARAMS ###
            self.eucl_norm_ = self.stopCondition()

            # Euclidean distance difference
            delta_norm = self.eucl_norm_ - eucl_norm_old

            # Loglikelihood difference
            # delta_ll = self.ll - ll_old

            '''SET STOP CONDITION (use once only)'''

            if np.abs(delta_norm) < self.min_change or i == self.max_it-1:

                # It does not work well, mainly for NTNames
                # self.printParams()

                # print("ll delta: {:.8} \nstop at {}th iteration\n".format(delta_ll, i+1))

                print("_____________________")
                print("euclidean delta: {:.8} \nstop at {}th iteration\n".format(np.abs(delta_norm), i+1))

                break

            else:
                eucl_norm_old = self.eucl_norm_
                # ll_old = self.ll

        return self

    def stopCondition(self):

        """Compute stop condition (loglikelihood and 2-norm)"""

        # 2-Norm
        eucl_norm = np.sum(distance.cdist(self.resp_, self.resp_, 'euclidean'))

        # Loglikelihood
        # ll = np.sum(np.log(self.likelihood))

        return eucl_norm

    def EStep(self, y):

        """To compute responsibilities, or posterior probability p(y/x)
        X = N X D matrix
        priors = C dimensional vector
        p = C X D matrix
        prob or resp (result) = N X C matrix"""

        # Step 1
        # Calculate the p(x/means)
        prob = self.distrBernoulli()

        # Step 2
        # Calculate the numerator of the resps
        num_resp = prob * self.priors_

        # Step 3
        # Calculate the denominator of the resps
        den_resp = num_resp.sum(axis=1)[:, np.newaxis]

        # Step 4
        # Calculate the responsibilities
        try:
            resp = num_resp/den_resp

            N = self.N_

            # I set the value "1" for known memberships
            for n in range(N):
                if y[n] == 0:
                    resp[n] = [1, 0]
                if y[n] == 1:
                    resp[n] = [0, 1]

            return resp, den_resp

        except ZeroDivisionError:
            print("Division by zero occured in reponsibility calculations!")

    def distrBernoulli(self):

        """To compute the probability of x for each Bernoulli distribution
        X = N X D matrix
        p = C X D matrix
        prob (result) = N X C matrix"""

        N = self.N_
        C = self.n_classes_

        prob = np.empty((N, C))

        for n in range(N):
            for c in range(C):
                prob[n, c] = np.dot(np.power(self.p_[c], self.X_[n]), np.power(1 - self.p_[c], 1 - self.X_[n]))

        return prob

    def MStep(self):

        """Re-estimate the parameters using the current responsibilities
        X = N X D matrix
        resp = N X C matrix
        return revised weights (C vector) and means (C X D matrix)"""

        N = self.N_
        D = self.D_
        C = self.n_classes_

        # Numerator priors
        N_c = np.sum(self.resp_, axis=0)

        p = np.empty((C, D))

        for c in range(C):
            try:
                # Sum is over N data points
                p[c] = np.sum(self.resp_[:, c][:, np.newaxis] * self.X_, axis=0)/N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        # Priors and p
        return N_c/N, p

    def printParams(self):

        table = [['Class', 'Priors', 'feat.1', 'feat.2', 'feat.3', 'feat.4', 'feat.5', 'feat.6', 'feat.7', 'feat.8', 'feat.9', 'feat.10', 'feat.11', 'feat.12', 'feat.13'],

                 ['C=0', self.priors_[0], self.p_[0, 0], self.p_[0, 1], self.p_[0, 2],
                  self.p_[0, 3], self.p_[0, 4], self.p_[0, 5], self.p_[0, 6], self.p_[0, 7],
                  self.p_[0, 8], self.p_[0, 9], self.p_[0, 10], self.p_[0, 11], self.p_[0, 12]],

                 ['C=1', self.priors_[1], self.p_[1, 0], self.p_[1, 1], self.p_[1, 2],
                  self.p_[1, 3], self.p_[1, 4], self.p_[1, 5], self.p_[1, 6], self.p_[1, 7],
                  self.p_[1, 8], self.p_[1, 9], self.p_[1, 10], self.p_[1, 11], self.p_[1, 12]]]

        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


    def fit_transform(self, X, y=None, **fit_params):

        self.fit(X, y)
        return self.transform()

    def transform(self):

        return self


    ################## PREDICT ##################

    def predict_proba(self, X):

        check_is_fitted(self)
        X = check_array(X)
        res = []

        N = X.shape[0]
        C = self.n_classes_

        for n in range(N):
            prob = []
            for c in range(C):
                prob.append((self.p_[c]**X[n] * (1-self.p_[c])**(1-X[n])).prod()*self.priors_[c])

            prob = np.array(prob)
            res.append(prob / prob.sum())

        return np.array(res)

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)


