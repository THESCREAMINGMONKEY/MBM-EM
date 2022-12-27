from random import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from scipy.spatial import distance

seed = 42
np.random.seed(seed)
rand = random()

class EMMB(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True


    def fit(self, X, y):

        """EM algorithm of Multivariate of Bernoulli Distributions"""

        X, y = check_X_y(X, y)
        #self.X = np.full((X.shape[0], X.shape[1]), 0.5)
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_)

        '''INIT PARAMS'''

        ### INIT PRIORS ###
        # Init with uniform random
        self.priors = np.random.uniform(.25, .75, self.n_classes_)
        tot = np.sum(self.priors)
        self.priors = self.priors/tot

        # Init with counts
        '''unique, counts = np.unique(y, return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        self.priors = [c0 / (c1 + c0), c1 / (c1 + c0)]'''

        ### INIT P ###
        self.p = np.random.uniform(size=(self.n_classes_, self.D))

        ### E STEP (INIT) ###
        self.resp, self.likelihood = self.EStep()

        ### SET STOP CONDITION PARAMS (INIT) ###
        eucl_norm_old, ll_old = self.stopCondition()

        '''############################################################################
           ############################### EM ALGO LOOP ###############################
           ############################################################################'''

        for i in range(self.max_it):

            ### M STEP ###
            self.priors, self.p = self.MStep()

            ### E STEP ###
            self.resp, self.likelihood = self.EStep()

            ### SET STOP CONDITION PARAMS ###
            self.eucl_norm, self.ll = self.stopCondition()

            # Euclidean distance difference
            delta_norm = self.eucl_norm - eucl_norm_old

            # Loglikelihood difference<
            delta_ll = self.ll - ll_old

            '''SET STOP CONDITION (use once only)'''

            #if np.abs(delta_ll) < self.min_change:
            if i == self.max_it-1:

                print("priors\n", self.priors)
                print("p[c=0]\n", self.p[0,:])
                print("p[c=1]\n", self.p[1,:])

                #print("ll delta: {:.8} \nstop at {}th iteration\n".format(delta_ll, i+1))
                #print("euclidean delta: {:.8} \nstop at {}th iteration\n".format(delta_norm, i+1))
                print("_____________________")

                break

            else:
                eucl_norm_old = self.eucl_norm
                ll_old = self.ll

        return self

    def stopCondition(self):

        """Compute stop condition (loglikelihood and 2-norm)"""

        eucl_norm = np.sum(distance.cdist(self.resp, self.resp, 'euclidean'))
        ll = np.sum(np.log(self.likelihood))

        return eucl_norm, ll

    def EStep(self):

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

                # OLD - SAME RESULTS OF THE DISTR. COMPUTED IN PREDICT PROBA
                prob[n, c] = np.prod((self.p[c]**self.X[n]) * ((1-self.p[c])**(1-self.X[n])))

                # SAME IN PREDICT PROBA
                #prob[n, c] = (self.p[c]*self.X[n] + (1-self.p[c])*(1-self.X[n])).prod()

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
        Nc = np.sum(self.resp, axis=0)
        mus = np.empty((C, D))

        for c in range(C):
            mus[c] = np.sum(self.resp[:, c][:, np.newaxis] * self.X, axis=0) #sum is over N data points

            try:
                mus[c] = mus[c]/Nc[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        #priors and p
        return Nc/N, mus

    ################## PREDICT ##################

    def predict_proba(self, X):

        check_is_fitted(self)
        X = check_array(X)
        res = []

        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):

                #print(self.priors_[j])
                #print(self.p_[j])
                #print(X[i])

                #SAME IN DISTR. BERNOULLI
                #probas.append((self.p[j]**self.X[i] * (1-self.p[j])**(1-self.X[i])).prod()*self.priors[j])

                #OLD
                probas.append((self.p[j]*self.X[i] + (1-self.p[j])*(1-self.X[i])).prod()*self.priors[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())

        return np.array(res)

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)