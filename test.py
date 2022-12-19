import numpy as np
import matplotlib.pyplot as plt
from numpy import size

plt.style.use('dark_background')

# Parameters for the simulation.
seed = 4
np.random.seed(seed)
K = 3
D = 10
N = 10000
max_it = 100
min_change = 0.1

# Model parameters.
true_priors_ = np.array([[0.1, 0.6, 0.3]])
true_p_ = np.random.uniform(size=(K, D))
X = np.zeros(shape=(N, D))

# Populating the data.
for n in range(N):
    k = np.random.choice(3, p=true_priors_[0])
    X[n, ] = np.random.binomial(n=1 , p=true_p_[k, ])


# Defining a class for the EM algorithm.
class MMB_EM:

    def __init__(self, max_it, min_change, verbose=True):
        """A Class that will hold information related with a expectation
        maximization algorithm when assuming a mixture of multivariate
        bernoulli distributions as the data generator process.

        Parameters
        ----------
        max_it : int
            Number of maximum iterations to run the simulation.
        min_change : float
            Minimum change of the log likelihood per iteration to continue
            the fitting process.
        verbose : float, default True
            Whether to print or not information about the training.

        """
        # Attributes.
        self.X = None
        self.max_it = max_it
        self.min_change = min_change
        self.N = None
        self.D = None
        self.verbose = verbose
        self.K = None
        self.priors_ = None
        self.p_ = None
        self.ll = None


    def fit(self, X, K=3):
        """Fit the desired data X with a mixture of K
        multivariate bernoulli distributions using the
        Expected Maximization Algorithm.

        Parameters
        ----------
        X : np.array
            Numpy array that will hold the data, it must be filled with ones
            or zeroes.
        K : int
            Number of multivariate bernoulli distributions.

        Return
        ------
        None

        """
        # Number of multivariate bernoulli distributions.
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.K = K
        self.ll = []

        print("SIZE K")
        print(size(self.K))


        # Populating randomly the parameters to estimate.
        self.priors_ = np.array([[1 / K for k in range(self.K)]])
        self.p_ = np.random.uniform(size=(self.K, self.D))

        print("MBMEM PRIORS E P")
        print(self.priors_.shape)
        print(self.priors_.size)
        print(self.priors_)

        print(self.p_.shape)
        print(self.p_.size)
        print(self.p_)
        print("X SHAPE")
        print(self.X.shape)


    # Training loop.
        for i in range(self.N):

            # E-step

            # p_x_p_ matrix shape (N, K), obs X_{n, k} given p_{k}
            # P(X_{n, k} | p_{k}).
            p_x_p_ = np.exp((np.dot(self.X, np.log(self.p_.T))) + (np.dot((1 - self.X), np.log((1 - self.p_.T)))))
            #p_x_p_ = np.exp(self.X @ np.log(self.p_.T) + (1 - self.X) @ np.log(1 - self.p_.T))

            # Calculating the resp P(Z | X, p_, priors).
            # Same shape / dim p_x_p_.
            # Numerator = priors_{k} * p(X_n | \p_k)
            # Denominator = sum_k \priors_{k} * p(X_n | \p_k).

            num_resp = p_x_p_ * self.priors_ #1000x3
            den_resp = np.sum(num_resp, axis=1, keepdims=True) #1000x1
            resp = num_resp / den_resp #1000x3

            # Calc ll

            self.ll.append(np.sum(np.log(den_resp)))

            # Stop condition
            if len(self.ll) > 1:
                delta = self.ll[-1] - self.ll[-2]

                if delta <= self.min_change:
                    return None

            # Printing the results.
            if self.verbose:
                print(f'Iteration: {i :4d} | Log likelihood: {self.ll[i] : 07.5f}')

            # M-step

            # Maximizes current Q(parameters).
            # priors_(t).
            num_priors = np.sum(resp, axis=0, keepdims=True) #1x3
            self.priors_ = num_priors / self.N #1x3

            # p_(t)
            self.p_ = (resp.T @ self.X) / num_priors.T #3x10
            print(self.p_.shape)
            print(self.p_.size)


    def plot(self, true_mu, true_pi):
        """Plot the results of the fitting procedure.

        Parameters
        ----------
        true_mu : np.array
            Numpy array containing the true values for mu.
        true_pi : np.array
            Numpy array containing the true values for pi.

        """
        # Verifying that the model has been fitted.
        assert self.K is not None, "The model hasn't being fitted"

        # Plotting the results.
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        # Plotting the mu's.
        axs[0, 0].set_title("True mu's")
        axs[0, 1].set_title("Fitted mu's")
        axs[1, 0].set_title("True and fitted pi's")
        axs[1, 0].set_xticks(range(2))
        axs[1, 0].set_xticklabels(("True", "Fitted"))
        axs[1, 1].set_title("Log likelihood")
        axs[1, 1].set_xlabel("Iteration")

        for i in range(self.K):
            # Calculating the bottom for the stacked bar plot.
            if i == 0:
                true_bottom = 0
                fitted_bottom = 0
                priors_bottom = 0
            else:
                true_bottom = np.sum(true_p_[0:i, ], axis=0)
                fitted_bottom = np.sum(self.p_[0:i, ], axis=0)
                priors_bottom = [np.sum(true_priors_[0, :i]), np.sum(self.priors_[0, :i])]

            color = plt.cm.viridis((i + 1) / self.K)

            axs[0, 0].bar(range(self.D),
                          true_p_[i, ],
                          bottom=true_bottom,
                          color=color)
            axs[0, 1].bar(range(self.D),
                          self.p_[i, ],
                          bottom=fitted_bottom,
                          color=color)
            axs[1, 0].bar(range(2),
                          [true_priors_[0, i], self.priors_[0, i]],
                          bottom=priors_bottom,
                          color=color)

        axs[1, 1].plot(range(len(self.ll)), self.ll)

        plt.show()


def main():
    model = MMB_EM(X, max_it, min_change)
    model.fit(X, K=3)

    model.plot(true_p_, true_priors_)

if __name__ == "__main__":
    main()