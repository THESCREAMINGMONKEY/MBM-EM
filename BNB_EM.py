import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from EM import EM

class BNB_EM(BaseEstimator, ClassifierMixin):

    def __init__(self, max_it, min_change, is_HBM):

        self.max_it = max_it
        self.min_change = min_change
        self.verbose = True
        self.is_HBM = is_HBM

    def fit(self, X, y):

        """EM algorithm of Multivariate of Bernoulli Distributions"""

        X, y = check_X_y(X, y)
        self.X_ = X
        self.N_ = X.shape[0] # Number of individuals
        self.D_ = X.shape[1] # Number of features

        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_) # Number of classes
        em = EM(X, X.shape[0], X.shape[1], len(self.classes_))

        if self.is_HBM == False: # if self.is_HBM == True:
                                 # these values are calculated in mixture.py -> check_X_train & check_X_test

            ## E-M FOR UNKNOWN VALUE IN X (x_i) ##
            self.X_ = em.EM_inc_values()


        # Create two arrays, one for unlabeled data and the other for the labeled
        labeled_X_, unlabeled_X_ = em.searchUnlabeled(y)

        # E STEP AND M STEP INIT ON ONLY LABELED DATA
        self.resp_, self.priors_, self.p_ = em.learnParams(y, labeled_X_)

        ### E STEP OF UNLABELED AND LABELED DATA ###
        self.resp_, self.likelihood_ = em.EStep(self.priors_, self.p_, y)

        ### SET STOP CONDITION PARAMS (INIT) ###
        eucl_norm_old = em.stopCondition(self.resp_)

        '''############################### EM ALGO LOOP ###############################'''

        for i in range(self.max_it):

            ### M STEP ###
            self.priors_, self.p_ = em.MStep(self.resp_)

            ### E STEP ###
            self.resp_, self.likelihood_ = em.EStep(self.priors_, self.p_, y)

            ### SET STOP CONDITION PARAMS ###
            self.eucl_norm_ = em.stopCondition(self.resp_)

            # Euclidean distance difference
            delta_norm = self.eucl_norm_ - eucl_norm_old

            # Loglikelihood difference
            # delta_ll = self.ll - ll_old

            '''SET STOP CONDITION (use once only)'''

            if np.abs(delta_norm) < self.min_change or i == self.max_it-1:

                # print("ll delta: {:.8} \nstop at {}th iteration\n".format(delta_ll, i+1))
                #print("_____________________")
                #print("euclidean delta: {:.8} \nstop at {}th iteration\n".format(np.abs(delta_norm), i+1))
                break

            else:
                eucl_norm_old = self.eucl_norm_
                # ll_old = self.ll

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