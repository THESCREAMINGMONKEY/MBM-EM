import warnings
from random import random

import numpy as np
import self
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import BernoulliNB, _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import _check_sample_weight, check_random_state


class BNB_EM(_BaseDiscreteNB, ClassifierMixin):

    def __init__(self):

        self.class_log_prior_ = random()
        self.feature_log_prob_ = random()
        self.log_resp = None
        self.n_init = 1
        self.random_state = None
        self.max_iter = 100
        self.alpha = 1
        self.binarize = None
        self.fit_prior = False
        self.class_prior = None


    def fit(self, X, y, sample_weight=None):

        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        '''class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)'''

        # --------------------------------------------------

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(self.n_init):
            for n_iter in range(1, self.max_iter + 1):

                log_resp = self._e_step(X)
                self._m_step(X, log_resp)

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        log_resp = self._e_step(X)

        return self.log_resp.argmax(axis=1)

        # return self

    # -----------------------------------------------------

    # M step

    def _m_step(self, X, log_resp):

        self._update_feature_prob(log_resp)
        self._update_class_prior(log_resp)
        pass


    def _update_class_prior(self, log_resp):

        self.log_resp = log_resp
        self.class_log_prior_ = (((self.log_resp/self.log_resp.sum())).sum()).mean(axis=0)

    def _update_feature_prob(self, log_resp):

        self.log_resp = log_resp
        self.feature_log_prob_ = (self.log_resp*self).sum()/self.log_resp.sum()


    # E step

    def _e_step(self, X):

        log_resp = self._joint_log_likelihood(X)

        return log_resp


    def _joint_log_likelihood(self, X):

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll

    # -----------------------------------------------------

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        pass