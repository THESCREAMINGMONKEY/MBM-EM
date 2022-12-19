from random import random

import numpy as np
from scipy.stats import bernoulli, stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from scipy.special import logsumexp

seed = 42
np.random.seed(seed)
rand = random()

class EMMB(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True


    def fit(self, X, y):

        '''
        EM algo of Multivariate of Bernoulli Distributions
        '''

        X, y = check_X_y(X, y)

        #self.X = np.full((X.shape[0], X.shape[1]), 0.5)
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.ll = 0
        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_)

        # INIT PARAMS #

        # priors init with uniform random
        self.priors = np.random.uniform(.25, .75, self.n_classes_)
        tot = np.sum(self.priors)
        self.priors = self.priors/tot


        # priors init with count
        '''unique, counts = np.unique(y, return_counts=True)
        unique = list(unique)
        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
        self.priors = [c0 / (c1 + c0), c1 / (c1 + c0)]'''

        self.p = np.random.uniform(size=(self.n_classes_, self.D))

        # E STEP INIT #

        init = True
        self.ll, self.resp = self.llBernoulli(init, y)
        ll_old = self.ll

        # EM ALGO #
        for i in range(self.max_it):
            #if self.verbose and (i % 5 == 0):

            '''
            print("iteration {}:".format(i))
            print("   {}:".format(self.priors))
            print("   {:.6}".format(self.ll))
            '''

            #M Step
            self.priors, self.p = self.bernoulliMStep()

            # E step and convergence check
            init = False
            self.ll, self.resp = self.llBernoulli(init, y)

            delta = self.ll-ll_old

            if np.abs(delta) < self.min_change:

                print("ll delta: {:.8} \nstop at {}th iteration \nfinal ll is {}\n".format(delta, i+1, self.ll))
                break
            else:

                ll_old = self.ll

        return self


    def llBernoulli(self, init, y):

        '''
        To compute expectation of the loglikelihood of Multivariate of Beroulli distributions.
        Since computing E(LL) requires computing responsibilities, this function does a double-duty
        to return responsibilities too
        '''

        N = self.N
        C = self.n_classes_

        resp, like = self.respBernoulli(init, y)

        #ll = logsumexp(like)
        ll = np.sum(np.log(like))

        return ll, resp


    def respBernoulli(self, init, y):

        '''
        To compute responsibilities, or posterior probability p(y/x)
        X = N X D matrix
        priors = Classes dimensional vector
        p = Classes X D matrix
        prob or resp (result) = N X Classes matrix
        '''

        #step 1
        # calculate the p(x/means)
        prob = self.bernoulli(init, y)

        #step 2
        # calculate the numerator of the resps
        num_resp = prob * self.priors

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


    def bernoulli(self, init, y):

        '''
        To compute the probability of x for each Bernoulli distribution
        X = N X D matrix
        p = Classes X D matrix
        prob (result) = N X classes matrix
        '''

        N = self.N
        C = self.n_classes_

        # Compute prob(x/p)

        prob = np.zeros((N, C))

        for n in range(N):
            for c in range(C):


                prob[n, c] = np.prod((self.p[c]**self.X[n][np.where(any!=0.5)]) *
                                     ((1-self.p[c])**(1-self.X[n][np.where(any!=0.5)])))
                
                #prob[n, c] = np.prod((self.p[c]**self.X[n]) * ((1-self.p[c])**(1-self.X[n])))'''

        # Try with init
        '''if init == True:
            for n in range(N):
                for c in range(C):

                    #prob[n, c] = np.prod((self.p[c]**self.X[n]) * ((1-self.p[c])**(1-self.X[n])))
                    prob[n, c] = np.prod((self.p[c]**self.X[n][np.where(any!=0.5 or y[n]!=-1)]) *
                                         ((1-self.p[c])**(1-self.X[n][np.where(any!=0.5 or y[n]!=-1)])))

        else:
            for n in range(N):
                for c in range(C):

                    prob[n, c] = np.prod((self.p[c]**self.X[n][np.where(any==0.5)]) *
                                         ((1-self.p[c])**(1-self.X[n][np.where(any==0.5)])))
                    #prob[n, c] = np.prod((self.p[c]**self.X[n][np.where(any!=0.5)]) * ((1-self.p[c])**(1-self.X[n][np.where(any!=0.5)])))'''

        return prob


    def bernoulliMStep(self):

        '''
        Re-estimate the parameters using the current responsibilities
        X = N X D matrix
        resp = N X Classes matrix
        return revised weights (C vector) and means (Classes X D matrix)
        '''

        N = self.N
        D = self.D
        C = len(self.resp[0])

        Nc = np.sum(self.resp, axis=0)
        mus = np.empty((C, D))

        for c in range(C):
            mus[c] = np.sum(self.resp[:, c][:, np.newaxis] * self.X, axis=0) #sum is over N data points
            try:
                
                mus[c] = mus[c]/Nc[c]
            except ZeroDivisionError:
                
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")
                break

        return Nc/N, mus


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

                probas.append((self.p[j]*X[i] +
                               (1-self.p[j])*(1-X[i])).prod()
                              *self.priors[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())


        return np.array(res)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)

