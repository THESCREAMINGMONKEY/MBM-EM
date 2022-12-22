from random import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from scipy.special import logsumexp
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

        """EM algo of Multivariate of Bernoulli Distributions"""


        X, y = check_X_y(X, y)

        #self.X = np.full((X.shape[0], X.shape[1]), 0)
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
        unique, counts = np.unique(y, return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        #self.priors = [c0 / (c1 + c0), c1 / (c1 + c0)]

        ### INIT P ###
        self.p = np.random.uniform(size=(self.n_classes_, self.D))

        ### E STEP INIT ###
        self.ll, self.resp = self.llBernoulli()

        ### SET STOP CONDITION PARAMS INIT ###
        ll_old = self.ll
        dist_old = distance.cdist(self.resp, self.resp, 'euclidean')

        '''#############################################################################
           ################################## EM ALGO ##################################
           #############################################################################'''

        for i in range(self.max_it):

            ### M STEP ###
            self.priors, self.p = self.bernoulliMStep()

            ### E STEP ###
            self.ll, self.resp = self.llBernoulli()

            ### SET STOP CONDITION PARAMS ###

            # Calc new Euclidean distance
            dist_new = distance.cdist(self.resp, self.resp, 'euclidean')

            # Euclidean distance difference
            delta_dist = np.sum(dist_new) - np.sum(dist_old)

            # Loglikelihood difference
            delta_ll = self.ll - ll_old

            '''SET STOP CONDITION (use once only)'''

            if np.abs(delta_ll) < self.min_change:
                print("priors\n", self.priors)
                print("p\n", self.p)

                print("ll delta: {:.8} \nstop at {}th iteration\n".format(delta_ll, i+1))
                #print("euclidean delta: {:.8} \nstop at {}th iteration\n".format(delta_dist, i+1))

                return self

            else:
                dist_old = distance.cdist(self.resp, self.resp, 'euclidean')
                ll_old = self.ll


    def llBernoulli(self):

            """To compute expectation of the loglikelihood of Multivariate of Bernoulli distributions.
            Since computing E(LL) requires computing responsibilities, this function does a double-duty
            to return responsibilities too"""


            N = self.N
            C = self.n_classes_

            resp, like = self.respBernoulli()

            #ll = logsumexp(like)
            ll = np.sum(np.log(like))

            return ll, resp


    def respBernoulli(self):

        """To compute responsibilities, or posterior probability p(y/x)
        X = N X D matrix
        priors = Classes dimensional vector
        p = Classes X D matrix
        prob or resp (result) = N X Classes matrix"""


        #step 1
        # calculate the p(x/means)
        #prob = self.bernoulli()

        #step 2
        # calculate the numerator of the resps
        #num_resp = prob * self.priors
        num_resp = self.bernoulli()

        #step 3
        # calcualte the denominator of the resp.s
        den_resp = num_resp.sum(axis=1)[:, np.newaxis]

        # step 4
        # calculate the resps
        try:
            resp = num_resp/den_resp
            return resp, den_resp

        except ZeroDivisionError:
            print("Division by zero occured in reponsibility calculations!")


    def bernoulli(self):

        """To compute the probability of x for each Bernoulli distribution
        X = N X D matrix
        p = Classes X D matrix
        prob (result) = N X classes matrix"""


        N = self.N
        C = self.n_classes_


        # Compute prob(x/p)

        prob = np.zeros((N, C))

        for n in range(N):
            for c in range(C):

                #prob[n, c] = np.prod((self.p[c]**self.X[n]) * ((1-self.p[c])**(1-self.X[n])))
                #prob[n, c] = np.dot((self.p[c]**self.X[n]), ((1-self.p[c])**(1-self.X[n])))

                #SAME IN PREDICT
                prob[n, c] = (self.p[c]*self.X[n] + (1-self.p[c])*(1-self.X[n])).prod()*self.priors[c]


                #prob[n, c] = np.dot(self.X[n], np.log(self.p[c])) + (np.dot((1 - self.X[n]), np.log((1 - self.p[c])))) #700x2

        return prob


    def bernoulliMStep(self):

        """Re-estimate the parameters using the current responsibilities
        X = N X D matrix
        resp = N X Classes matrix
        return revised weights (C vector) and means (Classes X D matrix)"""


        N = self.N
        D = self.D
        C = self.n_classes_

        Nc = np.sum(self.resp, axis=0)
        mus = np.empty((C, D))

        for c in range(C):
            mus[c] = np.sum(self.resp[:, c][:, np.newaxis] * self.X, axis=0) #sum is over N data points
            #mus[c] = np.sum(self.resp[:, c][:, np.newaxis] * self.X, axis=0) #sum is over N data points
            try:
                
                mus[c] = mus[c]/Nc[c]
            except ZeroDivisionError:
                
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")
                break

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

                #prob[n, c] = (self.p[j]*self.X[i] + (1-self.p[j])*(1-self.X[i])).prod()


                probas.append((self.p[j]*self.X[i] + (1-self.p[j])*(1-self.X[i])).prod()
                              *self.priors[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())


        return np.array(res)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)