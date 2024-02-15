# Work with Bayesian Linear Regression models with varying basis functions (linear, polynomial and Gaussian).
# Datasets used are 1D toy regression samples ranging from linear datasets
# to more complex non-linear datasets such as increasing sinusoidal curves.

# import libs
import numpy as np
import matplotlib.pyplot as plt

# Useful function: plot results
def plot_results(X_train, y_train, X_test, y_test, y_pred, std_pred,
                 xmin=-2, xmax=2, ymin=-2, ymax=1, stdmin=0.30, stdmax=0.45):
    """Given a dataset and predictions on test set, this function draw 2 subplots:
    - left plot compares train set, ground-truth (test set) and predictions
    - right plot represents the predictive variance over input range

    Args:
      X_train: (array) train inputs, sized [N,]
      y_train: (array) train labels, sized [N, ]
      X_test: (array) test inputs, sized [N,]
      y_test: (array) test labels, sized [N, ]
      y_pred: (array) mean prediction, sized [N, ]
      std_pred: (array) std prediction, sized [N, ]
      xmin: (float) min value for x-axis on left and right plot
      xmax: (float) max value for x-axis on left and right plot
      ymin: (float) min value for y-axis on left plot
      ymax: (float) max value for y-axis on left plot
      stdmin: (float) min value for y-axis on right plot
      stdmax: (float) max value for y-axis on right plot

    Returns:
      None
    """
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.xlim(xmin = xmin, xmax = xmax)
    plt.ylim(ymin = ymin, ymax = ymax)
    plt.plot(X_test, y_test, color='green', linewidth=2,
             label="Ground Truth")
    plt.plot(X_train, y_train, 'o', color='blue', label='Training points')
    plt.plot(X_test, y_pred, color='red', label="BLR Poly")
    plt.fill_between(X_test, y_pred-std_pred, y_pred+std_pred, color='indianred', label='1 std. int.')
    plt.fill_between(X_test, y_pred-std_pred*2, y_pred-std_pred, color='lightcoral')
    plt.fill_between(X_test, y_pred+std_pred*1, y_pred+std_pred*2, color='lightcoral', label='2 std. int.')
    plt.fill_between(X_test, y_pred-std_pred*3, y_pred-std_pred*2, color='mistyrose')
    plt.fill_between(X_test, y_pred+std_pred*2, y_pred+std_pred*3, color='mistyrose', label='3 std. int.')
    plt.legend()

    plt.subplot(122)
    plt.title("Predictive variance along x-axis")
    plt.xlim(xmin = xmin, xmax = xmax)
    plt.ylim(ymin = stdmin, ymax = stdmax)
    plt.plot(X_test, std_pred**2, color='red', label="\u03C3² {}".format("Pred"))

    # Get training domain
    training_domain = []
    current_min = sorted(X_train)[0]
    for i, elem in enumerate(sorted(X_train)):
        if elem-sorted(X_train)[i-1]>1:
            training_domain.append([current_min,sorted(X_train)[i-1]])
            current_min = elem
    training_domain.append([current_min, sorted(X_train)[-1]])

    # Plot domain
    for j, (min_domain, max_domain) in enumerate(training_domain):
        plt.axvspan(min_domain, max_domain, alpha=0.5, color='gray', label="Training area" if j==0 else '')
    plt.axvline(X_train.mean(), linestyle='--', label="Training barycentre")

    plt.legend()
    plt.show()

# Part I: Linear Basis function model
# We start with a linear dataset where we will analyze the behavior of linear basis functions
# in the framework of Bayesian Linear Regression.

# Given hyperparameters for linear model

ALPHA = 2.0  # Prior precision
SIG = 0.2    # Noise standard deviation in the data
BETA = 1 / (2.0 * SIG ** 2)  # Likelihood precision
NB_POINTS =25


# Generate linear toy dataset
def f_linear(x, noise_amount, sigma):
    y = -0.3 + 0.5*x
    noise = np.random.normal(0, sigma, len(x))
    return y + noise_amount*noise

# Create training and test points
dataset_linear = {}
dataset_linear['X_train'] = np.random.uniform(0, 2, NB_POINTS)
dataset_linear['y_train'] = f_linear(dataset_linear['X_train'], noise_amount=1, sigma=SIG)
dataset_linear['X_test'] = np.linspace(-10,10, 10*NB_POINTS)
dataset_linear['y_test'] = f_linear(dataset_linear['X_test'], noise_amount=0, sigma=SIG)
dataset_linear['ALPHA'] = ALPHA
dataset_linear['BETA'] = 1/(2.0*SIG**2)

# Plot dataset
plt.figure(figsize=(7,5))
plt.xlim(xmax = 3, xmin =-1)
plt.ylim(ymax = 1.5, ymin = -1)
plt.plot(dataset_linear['X_test'], dataset_linear['y_test'], color='green', linewidth=2, label="Ground Truth")
plt.plot(dataset_linear['X_train'], dataset_linear['y_train'], 'o', color='blue', label='Training points')
plt.legend()
plt.show()

# We will use the linear basis function:

def phi_linear(x):
    return np.array((1, x))



# Function to compute design matrix
def design_matrix(X):
    return np.vstack((np.ones(X.shape), X)).T

# Function to compute posterior mean and covariance
def compute_posterior(X, y, alpha, beta):
    Phi = design_matrix(X)
    Sigma_inv = alpha * np.eye(2) + beta * np.dot(Phi.T, Phi)
    Sigma = np.linalg.inv(Sigma_inv)
    mu = beta * np.dot(np.dot(Sigma, Phi.T), y)
    return mu, Sigma

plt.figure(figsize=(15,7))
for count, n in enumerate([0, 1, 2, 10, len(dataset_linear['X_train'])]):
    cur_data = dataset_linear['X_train'][:n]
    cur_lbl = dataset_linear['y_train'][:n]
    if n == 0:
        # Special case for zero data points (prior)
        mu = np.zeros(2)
        Sigma = np.linalg.inv(ALPHA * np.eye(2))
    else:
        mu, Sigma = compute_posterior(cur_data, cur_lbl, ALPHA, BETA)

    sigmainv = np.linalg.inv(Sigma)
    meshgrid = np.arange(-1, 1.01, 0.01)
    w = np.zeros((2,1))
    posterior = np.zeros((meshgrid.shape[0], meshgrid.shape[0]))

    # Compute values on meshgrid
    for i in range(meshgrid.shape[0]):
        for j in range(meshgrid.shape[0]):
            w[0, 0] = meshgrid[i]
            w[1, 0] = meshgrid[j]
            posterior[i, j] = np.exp(-0.5 * np.dot(np.dot((w - mu.reshape(2,1)).T, sigmainv), (w - mu.reshape(2,1))))
    Z = 1.0 / (np.sqrt(2 * np.pi * np.linalg.det(Sigma)))
    posterior /= Z

    # Plot posterior with n points
    plt.subplot(151 + count)
    plt.imshow(posterior, extent=[-1, 1, -1, 1], origin='lower')
    plt.plot(0.5, 0.3, '+', markeredgecolor='white', markeredgewidth=3, markersize=12)
    plt.title(f'Posterior with N={n} points')

plt.tight_layout()
plt.show()

def closed_form(func, X_train, y_train, alpha, beta):
    """Define analytical solution to Bayesian Linear Regression, with respect to the basis function chosen, the
    training set (X_train, y_train) and the noise precision parameter beta and prior precision parameter alpha chosen.
    It should return a function outputing both mean and std of the predictive distribution at a point x*.

    Args:
      func: (function) the basis function used
      X_train: (array) train inputs, size (N,)
      y_train: (array) train labels, size (N,)
      alpha: (float) prior precision parameter
      beta: (float) noise precision parameter

    Returns:
      (function) prediction function, returning itself both mean and std
    """

    # Compute the design matrix using the basis function
    Phi = np.array([func(x) for x in X_train])

    # Compute posterior covariance Sigma
    Sigma = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * np.dot(Phi.T, Phi))

    # Compute posterior mean mu
    mu = beta * np.dot(np.dot(Sigma, Phi.T), y_train)

    # Prediction function that computes mean and std of the predictive distribution
    def f_model(x):
        # Transform x* into feature space
        phi_x = np.array(func(x))

        # Compute mean of the predictive distribution
        mean = np.dot(mu.T, phi_x)

        # Compute variance (std^2) of the predictive distribution
        sigma2 = (1 / beta) + np.dot(np.dot(phi_x.T, Sigma), phi_x)

        # Return mean and standard deviation
        return mean, np.sqrt(sigma2)

    return f_model

# Initialize predictive function
f_pred = closed_form(phi_linear, dataset_linear['X_train'], dataset_linear['y_train'],
                     dataset_linear['ALPHA'], dataset_linear['BETA'])

# Assuming dataset_linear['X_test'] is defined and f_pred has been initialized as above
X_test = dataset_linear['X_test']
y_test = dataset_linear['y_test']  # Assuming you have the ground truth for test set for visualization

# Predictions and uncertainties for each point in the test set
y_pred = []
std_pred = []
for x in X_test:
    mean, std = f_pred(x)
    y_pred.append(mean)
    std_pred.append(std)

# Convert to numpy arrays for plotting
y_pred = np.array(y_pred)
std_pred = np.array(std_pred)

# Visualization using the specified parameters
plot_results(dataset_linear['X_train'], dataset_linear['y_train'], X_test, y_test, y_pred, std_pred,
             xmin=-10, xmax=10, ymin=-6, ymax=6, stdmin=0.05, stdmax=1)




# Generate dataset with a "hole"
X_train_hole = np.concatenate(([np.random.uniform(-3, -1, 10), np.random.uniform(1, 3, 10)]), axis=0)
y_train_hole = -0.3 + 0.5 * X_train_hole + np.random.normal(0, SIG, len(X_train_hole))
X_test_hole = np.linspace(-12, 12, 100)
y_test_hole = -0.3 + 0.5 * X_test_hole

# Plot dataset
plt.figure(figsize=(7,5))
plt.xlim(xmin =-12, xmax = 12)
plt.ylim(ymin = -7, ymax = 6)
plt.plot(X_test_hole, y_test_hole, color='green', linewidth=2, label="Ground Truth")
plt.plot(X_train_hole, y_train_hole, 'o', color='blue', label='Training points')
plt.legend()
plt.show()

def phi_linear(x):
    """Linear basis function."""
    return np.array([1, x])

def closed_form(func, X_train, y_train, alpha, beta):
    """Define analytical solution to Bayesian Linear Regression."""
    Phi = np.vstack([func(x) for x in X_train])
    Sigma = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * np.dot(Phi.T, Phi))
    mu = beta * np.dot(Sigma, np.dot(Phi.T, y_train))

    def predict(x):
        phi_x = func(x)
        mean = np.dot(phi_x, mu)
        sigma = 1 / beta + np.dot(phi_x, np.dot(Sigma, phi_x))
        return mean, np.sqrt(sigma)

    return predict

# Define prediction function for the new dataset
f_pred_hole = closed_form(phi_linear, X_train_hole, y_train_hole, ALPHA, BETA)

# Predict on test points
y_pred_hole = np.array([f_pred_hole(x)[0] for x in X_test_hole])
std_pred_hole = np.array([f_pred_hole(x)[1] for x in X_test_hole])

# Visualization function adapted for the current context
def plot_results_hole(X_train, y_train, X_test, y_test, y_pred, std_pred, xmin, xmax, ymin, ymax, stdmin, stdmax):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.plot(X_test, y_test, color='green', linewidth=2, label="Ground Truth")
    plt.plot(X_train, y_train, 'o', color='blue', label='Training points')
    plt.plot(X_test, y_pred, color='red', label="Predictions")
    plt.fill_between(X_test, y_pred - std_pred, y_pred + std_pred, color='salmon', alpha=0.5, label='Predictive std. dev.')
    plt.legend()

    plt.subplot(122)
    plt.title("Predictive variance along x-axis")
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=stdmin, ymax=stdmax)
    plt.plot(X_test, std_pred ** 2, color='red', label="Variance")
    plt.legend()
    plt.show()

# Plot results
plot_results_hole(X_train_hole, y_train_hole, X_test_hole, y_test_hole, y_pred_hole, std_pred_hole,
                  xmin=-12, xmax=12, ymin=-7, ymax=6, stdmin=0.0, stdmax=0.5)

