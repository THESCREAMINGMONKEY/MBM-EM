# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from scipy.special import psi,gammaln
from scipy.special import logsumexp
from scipy.linalg import pinvh
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix,isspmatrix
from sklearn.cluster import KMeans
import numpy as np
from EM import EM




#============================= Helpers =============================================#


class StudentMultivariate(object):
    '''
    Multivariate Student Distribution
    '''
    def __init__(self,mean,precision,df,d):
        self.mu   = mean
        self.L    = precision
        self.df   = df
        self.d    = d


    def logpdf(self,x):
        '''
        Calculates value of logpdf at point x
        '''
        xdiff     = x - self.mu
        quad_form = np.sum( np.dot(xdiff,self.L)*xdiff, axis = 1)


        return ( gammaln( 0.5 * (self.df + self.d)) - gammaln( 0.5 * self.df ) +
                 0.5 * np.linalg.slogdet(self.L)[1] - 0.5*self.d*np.log( self.df*np.pi) -
                 0.5 * (self.df + self.d) * np.log( 1 + quad_form / self.df )
                 )

    def pdf(self,x):
        '''
        Calculates value of pdf at point x
        '''
        return np.exp(self.logpdf(x))



def _e_log_dirichlet(alpha0,alphaK):
    ''' Calculates expectation of log pdf of dirichlet distributed parameter '''
    log_C   = gammaln(np.sum(alpha0)) - np.sum(gammaln(alpha0))
    e_log_x = np.dot(alpha0-1,psi(alphaK) - psi(np.sum(alphaK)))
    return np.sum(log_C + e_log_x)


def _e_log_beta(c0,d0,c,d):
    ''' Calculates expectation of log pdf of beta distributed parameter'''
    log_C    = gammaln(c0 + d0) - gammaln(c0) - gammaln(d0)
    psi_cd   = psi(c+d)
    log_mu   = (c0 - 1) * ( psi(c) - psi_cd )
    log_i_mu = (d0 - 1) * ( psi(d) - psi_cd )
    return np.sum(log_C + log_mu + log_i_mu)


def _get_classes(X):
    '''Finds number of unique elements in matrix'''
    if isspmatrix(X):
        v = X.data
        if len(v) < X.shape[0]*X.shape[1]:
            v = np.hstack((v,np.zeros(1)))
            V = np.unique(v)
    else:
        V = np.unique(X)
    return V


#==================================================================================#


class GeneralMixtureModelExponential(BaseEstimator):
    '''
    Superclass for Mixture Models
    '''
    def __init__(self, n_components = 2, n_iter = 100, tol = 1e-3,
                 alpha0 = 10, n_init = 3, init_params = None,
                 compute_score = False, verbose = False):
        self.n_iter              = n_iter
        self.n_init              = n_init
        self.n_components        = n_components
        self.tol                 = tol
        self.alpha0              = alpha0
        self.compute_score       = compute_score
        self.init_params         = init_params
        self.verbose             = verbose


    def _update_resps(self, X, alphaK, *args):
        '''
        Updates distribution of latent variable with Dirichlet prior
        '''
        e_log_weights = psi(alphaK) - psi(np.sum(alphaK))
        return self._update_resps_parametric(X,e_log_weights,self.n_components,
                                             *args)


    def _update_resps_parametric(self, X, log_weights, clusters, *args):
        ''' Updates distribution of latent variable with parametric weights'''
        log_resps  = np.asarray([self._update_logresp_cluster(X,k,log_weights,*args)
                                 for k in range(clusters)]).T
        log_like       = np.copy(log_resps)
        log_resps     -= logsumexp(log_resps, axis = 1, keepdims = True)
        resps          = np.exp(log_resps)
        delta_log_like = np.sum(resps*log_like) - np.sum(resps*log_resps)
        return resps, delta_log_like


    def _update_dirichlet_prior(self,alpha_init,Nk):
        '''
        For all models defined in this module prior for cluster distribution
        is Dirichlet, so all models will need to update parameters
        '''
        return alpha_init + Nk


    def _check_X(self,X):
        '''
        Checks validity of input for all mixture models
        '''
        X  = check_array(X, accept_sparse = ['csr'])
        # check that number of components is smaller or equal to number of samples
        if X.shape[0] < self.n_components:
            raise ValueError(('Number of components should not be larger than '
                              'number of samples'))

        return X


    def _check_convergence(self,metric_diff,n_params):
        ''' Checks convergence of mixture model'''
        convergence = metric_diff / n_params < self.tol
        if self.verbose and convergence:
            print("Algorithm converged")
        return convergence


    def predict(self,X):
        '''
        Predict cluster for test data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Data Matrix

        Returns
        -------
        : array, shape = (n_samples,) component memberships
           Cluster index
        '''
        return np.argmax(self.predict_proba(X),1)



    def score(self,X):
        '''
        Computes the log probability under the model

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point

        Returns
        -------
        logprob: array with shape [n_samples,]
            Log probabilities of each data point in X
        '''
        probs = self.predict_proba(X)
        return np.log(np.dot(probs,self.weights_))



#==================================================================================#


class VBBMM(GeneralMixtureModelExponential):
    '''
    Variational Bayesian Bernoulli Mixture Model

    Parameters
    ----------
    n_components : int, optional (DEFAULT = 2)
        Number of mixture components

    n_init :  int, optional (DEFAULT = 5)
        Number of restarts of algorithm

    n_iter : int, optional (DEFAULT = 100)
        Number of iterations of Mean Field Approximation Algorithm

    tol : float, optional (DEFAULT = 1e-3)
        Convergence threshold

    alpha0 :float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on weights

    c : float , optional (DEFAULT = 1)
        Shape parameter for beta distribution

    d: float , optional (DEFAULT = 1)
        Shape parameter for beta distribution

    compute_score: bool, optional (DEFAULT = True)
        If True computes logarithm of lower bound at each iteration

    verbose : bool, optional (DEFAULT = False)
        Enable verbose output


    Attributes
    ----------
    weights_ : numpy array of size (n_components,)
        Mixing probabilities for each cluster

    means_ : numpy array of size (n_features, n_components)
        Mean success probabilities for each cluster

    scores_: list of unknown size (depends on number of iterations)
        Log of lower bound

    '''
    def __init__(self, n_components = 2, n_init = 10, n_iter = 100, tol = 1e-3,
                 alpha0 = 1, c = 1e-2, d = 1e-2, init_params = None,
                 compute_score = False, verbose = False):
        super(VBBMM,self).__init__(n_components,n_iter,tol,alpha0, n_init,
                                   init_params, compute_score, verbose)
        self.c = c
        self.d = d


    def _check_X_train(self, X):
        ''' Preprocesses & check validity of training data'''

        classes_ = [0, 1]
        em = EM(X, X.shape[0], X.shape[1], len(classes_))
        X = em.EM_inc_values() # compute unknow features for inds

        X                 = super(VBBMM,self)._check_X(X)
        self.classes_     = _get_classes(X)
        n                 = len(self.classes_)
        # check that there are only two categories in data
        if n != 2:
            raise ValueError(('There are {0} categorical values in data, '
                              'model accepts data with only 2'.format(n)))
        return 1*(X==self.classes_[1])


    def _check_X_test(self, X):
        ''' Preprocesses & check validity of test data'''

        classes_ = [0, 1]
        em = EM(X, X.shape[0], X.shape[1], len(classes_))
        X = em.EM_inc_values() # compute unknow features for inds

        X = check_array(X, accept_sparse = ['csr'])
        classes_   = _get_classes(X)
        n          = len(classes_)
        # check number of classes
        if n != 2:
            raise ValueError(('There are {0} categorical values in data, '
                              'model accepts data with only 2'.format(n)))
            # check whether these are the same classes as in training
        if classes_[0]==self.classes_[0] and classes_[1] == self.classes_[1]:
            return 1*(X==self.classes_[1])
        else:
            raise ValueError(('Classes in training and test set are different, '
                              '{0} in training, {1} in test'.format(self.classes_,
                                                                    classes_)))


    def _fit(self, X):
        '''
        Performs single run of VBBMM
        '''
        n_samples, n_features = X.shape
        n_params              = n_features*self.n_components + self.n_components
        scores                = []

        # use initial values of hyperparameter as starting point
        c = self.c * np.random.random([n_features,self.n_components])
        d = self.d * np.random.random([n_features,self.n_components])
        c_old, d_old = c,d
        c_prev,d_prev = c,d

        # we need to break symmetry for mixture weights
        alphaK      = self.alpha0*np.random.random(self.n_components)
        alphaK_old  = alphaK
        alphaK_prev = alphaK

        for i in range(self.n_iter):

            # ---- update approximating distribution of latent variable ----- #

            resps, delta_log_like = self._update_resps(X,alphaK,c,d)

            # reuse responsibilities in computing lower bound
            if self.compute_score:
                scores.append(self._compute_score(delta_log_like, alphaK_old,
                                                  alphaK, c_old, d_old, c, d))

            # ---- update approximating distribution of parameters ---------- #

            Nk     = sum(resps,0)

            # update parameters of Dirichlet Prior
            alphaK = self._update_dirichlet_prior(alphaK_old,Nk)

            # update parameters of Beta distributed success probabilities
            c,d    = self._update_params( X, Nk, resps)
            diff   = np.sum(abs(c-c_prev) + abs(d-d_prev) + abs(alphaK-alphaK_prev))

            if self.verbose:
                if self.compute_score:
                    print('Iteration {0}, value of lower bound is {1}'.format(i,scores[-1]))
                else:
                    print(('Iteration {0}, normalised delta of parameters '
                           'is {1}').format(i,diff))

            if self._check_convergence(diff,n_params):
                break
            c_prev,d_prev = c,d
            alphaK_prev   = alphaK

        # compute log of lower bound to compare best model
        resps, delta_log_like = self._update_resps(X,alphaK,c,d)
        scores.append(self._compute_score(delta_log_like, alphaK_old,
                                          alphaK, c_old, d_old, c, d))
        return alphaK, c, d, scores


    def _update_logresp_cluster(self,X,k,e_log_weights,*args):
        '''
        Unnormalised responsibilities for single cluster
        '''
        c,d   = args
        ck,dk = c[:,k], d[:,k]
        xcd   = safe_sparse_dot(X , (psi(ck) - psi(dk)))
        log_resp = xcd + np.sum(psi(dk) - psi(ck + dk)) + e_log_weights[k]
        return log_resp


    def _update_params(self,X,Nk,resps):
        '''
        Update parameters of prior distribution for Bernoulli Succes Probabilities
        '''
        XR = safe_sparse_dot(X.T,resps)
        c  = self.c + XR
        d  = self.d + (Nk - XR)
        return c,d


    def _compute_score(self, delta_log_like, alpha_init, alphaK, c_old, d_old, c, d):
        '''
        Computes lower bound
        '''
        log_weights_prior   =  _e_log_dirichlet(alpha_init, alphaK)
        log_success_prior   =  _e_log_beta(c_old,d_old,c,d)
        log_weights_approx  = -_e_log_dirichlet(alphaK,alphaK)
        log_success_approx  = -_e_log_beta(c,d,c,d)
        lower_bound         =  log_weights_prior
        lower_bound        +=  log_success_prior   + log_weights_approx
        lower_bound        +=  log_success_approx  + delta_log_like
        return lower_bound


    def fit(self, X):
        '''
        Fits Variational Bayesian Bernoulli Mixture Model

        Parameters
        ----------
        X: array-like or sparse csr_matrix of size [n_samples, n_features]
           Data Matrix

        Returns
        -------
        self: object
           self

        Practical Advice
        ----------------
        Significant speedup can be achieved by using sparse matrices
        (see scipy.sparse.csr_matrix)

        '''


        # preprocess data
        X = self._check_X_train(X)


        # refit & choose best model (log of lower bound is used)
        score_old        = [np.NINF]
        alpha_, c_ , d_  = 0,0,0
        for j in range(self.n_init):
            if self.verbose:
                print("New Initialisation, restart number {0} \n".format(j))

            alphaK, c, d, score = self._fit(X)
            if score[-1] > score_old[-1]:
                alpha_ , c_ , d_ = alphaK, c, d
                score_old        = score

        # save parameters corresponding to best model
        self.alpha_      = alpha_
        self.means_      = c_ / (c_ + d_)
        self.c_, self.d_ = c_,d_
        self.weights_    = alpha_ / np.sum(alpha_)
        self.scores_     = score_old
        return self

    def predict_proba(self,X):
        '''
        Predict probability of cluster for test data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data Matrix for test data

        Returns
        -------
        probs : array, shape = (n_samples,n_components)
            Probabilities of components membership
        '''
        check_is_fitted(self,'scores_')
        X = self._check_X_test(X)
        probs = self._update_resps(X,self.alpha_,self.c_, self.d_)[0]
        return probs


    def cluster_prototype(self):
        '''
        Computes most likely prototype for each cluster, i.e. vector that has
        highest probability of being observed under learned distribution
        parameters.

        Returns
        -------
        protoypes: numpy array of size (n_features,n_components)
           Cluster prototype
        '''
        prototypes = np.asarray([self.classes_[1*(self.means_[:,i] >=0.5)] for i in
                                 range(self.n_components)]).T
        return prototypes

    def transform(self, X):

        pred = self.predict(X)
        return np.reshape(pred, (len(pred), 1))

    def fit_transform(self, X, y): # need this y to implement pipeline

        self.fit(X)

        return self.transform(X)
