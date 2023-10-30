import numpy as np
from scipy.spatial import distance
import warnings

class EM():

    # FOR DIVIDE BY ZERO IN "EM_inc_values"
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    def __init__(self, X, n_samples, n_features, n_classes_):

        self.X_ = X
        self.N_ = n_samples
        self.D_ = n_features
        self.n_classes_ = n_classes_


    def EM_inc_values(self):

        """Implementation of the Expectation Maximization algorithm for imputing missing values in binary data.

         self.N_ = Number of samples
         self.D_ = Number of features
         Returns:
         imputed_data: The data matrix with imputed missing values (N X D matrix)"""

        num_iterations = 100
        tol = 0.0001
        prev_log_likelihood = 0

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

            log_likelihood = np.sum(np.nan_to_num(self.X_ * np.log(expected_value) + (1 - self.X_) * np.log(1 - expected_value)))

            if i > num_iterations or log_likelihood - prev_log_likelihood < tol:
                break
            else:
                prev_log_likelihood = log_likelihood

        self.X_ = imputed_data

        return imputed_data


    # ONLY FOR INITIALIZATION
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


    # ONLY FOR INITIALIZATION
    def learnParams(self, y, labeled_X_):

        """Use the labeled data to initialize params"""

        ### E STEP (INIT - RESP OF ONLY LABELED DATA [1 = C|c]) ###

        labeled_resp = []

        for n in range(self.N_):
            if y[n] == 0:
                labeled_resp.append([1,0])
            if y[n] == 1:
                labeled_resp.append([0,1])

        ### M STEP (INIT - LEARNING PARAMS FROM LABELED DATA) ###

        n_labeled_samples = labeled_X_.shape[0]
        labeled_resp_array = np.array(labeled_resp)

        # Numerator priors
        N_c = np.sum(labeled_resp_array, axis=0)
        p = np.empty((self.n_classes_, self.D_))

        for c in range(self.n_classes_):
            try:
                # Sum is over N data points
                p[c] = np.sum(labeled_resp_array[:, c][:, np.newaxis] * labeled_X_, axis=0) / N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")

                break

        return labeled_resp_array, N_c/n_labeled_samples, p


    def stopCondition(self, resp_):

        """Compute stop condition (loglikelihood and 2-norm)"""

        # 2-Norm
        eucl_norm = np.sum(distance.cdist(resp_, resp_, 'euclidean'))

        # Loglikelihood
        # ll = np.sum(np.log(self.likelihood))

        return eucl_norm


    def EStep(self, priors_, p_, y):

        """To compute responsibilities, or posterior probability p(y/x)
        X = N X D matrix
        priors = C dimensional vector
        p = C X D matrix
        prob or resp (result) = N X C matrix"""

        # Step 1
        # Calculate the p(x/means)
        prob = self.distrBernoulli(p_)

        # Step 2
        # Calculate the numerator of the resps
        num_resp = prob * priors_

        # Step 3
        # Calculate the denominator of the resps
        den_resp = num_resp.sum(axis=1)[:, np.newaxis]

        eps = 1e-3

        # Step 4
        # Calculate the responsibilities
        try:
            #resp = num_resp/den_resp
            resp = (num_resp+eps)/(den_resp+eps*num_resp.shape[0]) # Adj

            # I set the value "1" for known memberships
            for n in range(self.N_):
                if y[n] == 0:
                    resp[n] = [1, 0]
                if y[n] == 1:
                    resp[n] = [0, 1]

            return resp, den_resp

        except ZeroDivisionError:
            print("Division by zero occured in reponsibility calculations!")


    def distrBernoulli(self, p_):

        """To compute the probability of x for each Bernoulli distribution
        X = N X D matrix
        p = C X D matrix
        prob (result) = N X C matrix"""

        prob = np.empty((self.N_, self.n_classes_))

        for n in range(self.N_):
            for c in range(self.n_classes_):
                prob[n, c] = np.dot((p_[c] ** self.X_[n]), (1 - p_[c] ** (1 - self.X_[n])))
                #prob[n, c] = np.dot(np.power(self.p_[c], self.X_[n]), np.power(1 - self.p_[c], 1 - self.X_[n]))
                #prob[n, c] = np.dot(np.power(p_c[c], X_n), np.power(1 - p_c[c], (1 - X_n)))

        return prob


    def MStep(self, resp_):

        """Re-estimate the parameters using the current responsibilities
        X = N X D matrix
        resp = N X C matrix
        return revised weights (C vector) and means (C X D matrix)"""

        # Numerator priors
        N_c = np.sum(resp_, axis=0)
        p = np.empty((self.n_classes_, self.D_))

        for c in range(self.n_classes_):
            try:
                # Sum is over N data points
                p[c] = np.sum(resp_[:, c][:, np.newaxis] * self.X_, axis=0)/N_c[c]

            except ZeroDivisionError:
                print("Division by zero occured in Multivariate of Bernoulli Dist M-Step!")
                break

        # Priors and p
        return N_c/self.N_, p
