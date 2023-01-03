import numpy as np

from random import random
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from scipy.spatial import distance
from tabulate import tabulate

seed = 42
np.random.seed(seed)
rand = random()

class EMMB(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True


    def searchUnlabeled(self, y):

        """This function separate the labeled data from he unlabeled in two arrays"""

        N = self.N
        labeled_X = []
        unlabeled_X = []

        for n in range(N):
            if y[n] == 0:
                labeled_X.append(self.X[n])
            if y[n] == 1:
                labeled_X.append(self.X[n])
            if y[n] == -1:
                unlabeled_X.append(self.X[n])

        return np.array(labeled_X), np.array(unlabeled_X)

    def learnParams(self, y):

        """Use the labeled data to initialize params"""

        ### E STEP (INIT - RESP OF ONLY LABELED DATA [1 = C|c]) ###

        N = self.N
        labeled_resp = []

        for n in range(N):
            if y[n] == 0:
                labeled_resp.append([1,0])
            if y[n] == 1:
                labeled_resp.append([0,1])

        ### M STEP (INIT - LEARNING PARAMS FROM LABELED DATA) ###

        N = self.labeled_X.shape[0]
        D = self.D
        C = self.n_classes_
        labeled_resp_array = np.array(labeled_resp)

        # Numerator priors
        N_c = np.sum(labeled_resp_array, axis=0)

        p = np.empty((C, D))

        for c in range(C):
            try:
                # Sum is over N data points
                p[c] = np.sum(labeled_resp_array[:, c][:, np.newaxis] * self.labeled_X, axis=0) / N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        return labeled_resp_array, N_c/N, p

    def fit(self, X, y):

        """EM algorithm of Multivariate of Bernoulli Distributions"""

        X, y = check_X_y(X, y)
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_)

        # Create two arrays, one for unlabeled data and the other for the labeled
        self.labeled_X, self.unlabeled_X = self.searchUnlabeled(y)

        # E STEP AND M STEP INIT ON ONLY LABELED DATA
        self.resp, self.priors, self.p = self.learnParams(y)

        ### E STEP OF UNLABELED AND LABELED DATA ###
        self.resp, self.likelihood = self.EStep(y)

        ### SET STOP CONDITION PARAMS (INIT) ###
        eucl_norm_old, ll_old = self.stopCondition()

        '''############################################################################
           ############################### EM ALGO LOOP ###############################
           ############################################################################'''

        for i in range(self.max_it):

            ### M STEP ###
            self.priors, self.p = self.MStep()

            ### E STEP ###
            self.resp, self.likelihood = self.EStep(y)

            ### SET STOP CONDITION PARAMS ###
            self.eucl_norm, self.ll = self.stopCondition()

            # Euclidean distance difference
            delta_norm = self.eucl_norm - eucl_norm_old

            # Loglikelihood difference<
            delta_ll = self.ll - ll_old

            '''SET STOP CONDITION (use once only)'''

            if np.abs(delta_norm) < self.min_change or i == self.max_it-1:

                table = [['Class', 'Priors', 'feat.1', 'feat.2', 'feat.3', 'feat.4', 'feat.5', 'feat.6', 'feat.7', 'feat.8', 'feat.9', 'feat.10', 'feat.11', 'feat.12', 'feat.13'],

                         ['C=0', self.priors[0], self.p[0, 0], self.p[0, 1], self.p[0, 2],
                          self.p[0, 3], self.p[0, 4], self.p[0, 5], self.p[0, 6], self.p[0, 7],
                          self.p[0, 8], self.p[0, 9], self.p[0, 10], self.p[0, 11], self.p[0, 12]],

                         ['C=1', self.priors[1], self.p[1, 0], self.p[1, 1], self.p[1, 2],
                          self.p[1, 3], self.p[1, 4], self.p[1, 5], self.p[1, 6], self.p[1, 7],
                          self.p[1, 8], self.p[1, 9], self.p[1, 10], self.p[1, 11], self.p[1, 12]]]

                print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

                #print("ll delta: {:.8} \nstop at {}th iteration\n".format(delta_ll, i+1))
                print("euclidean delta: {:.8} \nstop at {}th iteration\n".format(np.abs(delta_norm), i+1))
                print("_____________________")

                break

            else:
                eucl_norm_old = self.eucl_norm
                ll_old = self.ll

        return self

    def stopCondition(self):

        """Compute stop condition (loglikelihood and 2-norm)"""

        # 2-Norm
        eucl_norm = np.sum(distance.cdist(self.resp, self.resp, 'euclidean'))

        # Loglikelihood
        ll = np.sum(np.log(self.likelihood))

        return eucl_norm, ll

    def EStep(self, y):

        """To compute responsibilities, or posterior probability p(y/x)
        X = N X D matrix
        priors = Classes dimensional vector
        p = Classes X D matrix
        prob or resp (result) = N X Classes matrix"""

        #step 1
        # Calculate the p(x/means)
        prob = self.distrBernoulli()

        #step 2
        # Calculate the numerator of the resps
        num_resp = prob * self.priors

        #step 3
        # Calculate the denominator of the resps
        den_resp = num_resp.sum(axis=1)[:, np.newaxis]

        # step 4
        # Calculate the resps
        try:
            resp = num_resp/den_resp

            N = self.N

            for n in range(N):
                if y[n] == 0:
                    resp[n] = [1,0]
                if y[n] == 1:
                    resp[n] = [0,1]

            return resp, den_resp

        except ZeroDivisionError:
            print("Division by zero occured in reponsibility calculations!")

    def distrBernoulli(self):

        """To compute the probability of x for each Bernoulli distribution
        X = N X D matrix
        p = Classes X D matrix
        prob (result) = N X classes matrix"""

        N = self.N
        C = self.n_classes_

        prob = np.empty((N, C))

        for n in range(N):
            for c in range(C):
                prob[n, c] = np.dot(np.power(self.p[c], self.X[n]), np.power(1 - self.p[c], 1 - self.X[n]))

        return prob

    def MStep(self):

        """Re-estimate the parameters using the current responsibilities
        X = N X D matrix
        resp = N X Classes matrix
        return revised weights (C vector) and means (Classes X D matrix)"""

        N = self.N
        D = self.D
        C = self.n_classes_

        # Numerator priors
        N_c = np.sum(self.resp, axis=0)

        p = np.empty((C, D))

        for c in range(C):
            try:
                # Sum is over N data points
                p[c] = np.sum(self.resp[:, c][:, np.newaxis] * self.X, axis=0)/N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        # Priors and p
        return N_c/N, p

    ################## PREDICT ##################

    def predict_proba(self, X):

        check_is_fitted(self)
        X = check_array(X)
        res = []

        N = X.shape[0]
        C = self.n_classes_

        for n in range(N):
            probas = []
            for c in range(C):
                probas.append((self.p[c]**self.X[n] * (1-self.p[c])**(1-self.X[n])).prod()*self.priors[c])

            probas = np.array(probas)
            res.append(probas / probas.sum())

        return np.array(res)

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)


