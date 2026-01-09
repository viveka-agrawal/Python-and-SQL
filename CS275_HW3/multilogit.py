
# UCI CS275P: Logistic Regression Classification

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt

# Imports to use for optimization
from functools import partial
import scipy.optimize

# Local imports
from classifier_utils import plotter_classifier, generate_features, \
							example_of_quadratic_use, example_of_plot_function_with_toy_dataset

def loss_func_logistic(w, x, t, K, alpha):
	''' Compute the L2-regularized logistic loss function 

		Arguments:
			w: estimated feature weight vector
			x: model matrix (NxM)
			t: response vector (Nx1)
			K: number of response classes
			alpha: regularization constant

		Returns:
			The loss function value
	'''
     
	# number of examples = N, number of features = M (for the NxM matrix)
	num_examples, num_features = x.shape  
    
	# convert one-hot to class indices (only if needed)
	if t.ndim > 1: t = np.argmax(t, axis=1)
    
    # reshape weight vector (# of features, # of response classes)
	w = w.reshape((num_features, K))  

	# calculate scores for each class (# of examples, # of response classes)
	scores = np.dot(x, w)  
    # subtract max score to prevent overflow during softmax computation
	scores -= np.max(scores, axis=1, keepdims=True) 

	# exponentiate scores to make all values positive
	exponential_scores = np.exp(scores)  
    # sum exponentiated scores across classes for each example (Nx1)
	sum_exponential_scores = np.sum(exponential_scores, axis=1, keepdims=True)
    # calculate softmax probabilities (# of examples, # of response classes)
	softmax_probability = exponential_scores / sum_exponential_scores  

    # obtain softmax probability of correct class for each example
	correct_class = softmax_probability[np.arange(num_examples), t]
    # obtain log probability for each sample
	log_probability = np.log(correct_class)
    # calculate negative log-likelihood
	log_likelihood = -np.sum(log_probability) 

	# calculate L2 regularization term
	reg_term = 0.5 * alpha * np.sum(w**2) 

	loss = log_likelihood + reg_term  
	return loss

def gradient_loss_func_logistic(w, x, t, K, alpha):
	''' Compute the gradient of the L2-regularized logistic loss (with respect to w) 

		Arguments:
			w: estimated feature weight vector
			x: model matrix (NxM)
			t: response vector (Nx1)
			K: number of response classes
			alpha: regularization constant

		Returns:
			The gradient of loss function with respect to w
	'''
     
	# number of examples = N, number of features = M (for the NxM matrix)
	num_examples, num_features = x.shape  
    
	# convert one-hot to class indices (only if needed)
	if t.ndim > 1: t = np.argmax(t, axis=1)
     
    # reshape weight vector (# of features, # of response classes)
	w = w.reshape((num_features, K))  

	# calculate scores for each class (# of examples, # of response classes)
	scores = np.dot(x, w)  
    # subtract max score to prevent overflow during softmax computation
	scores -= np.max(scores, axis=1, keepdims=True) 

	# exponentiate scores to make all values positive
	exponential_scores = np.exp(scores)  
    # sum exponentiated scores across classes for each example (Nx1)
	sum_exponential_scores = np.sum(exponential_scores, axis=1, keepdims=True)
    # calculate softmax probabilities (# of examples, # of response classes)
	softmax_probability = exponential_scores / sum_exponential_scores  
     
	# create one-hot encoded matrix (# of examples, # of response classes)
    # initialize matrix with zeros
	one_hot_matrix = np.zeros_like(softmax_probability)  
    # correct class indices for each example are set to 1
	one_hot_matrix[np.arange(num_examples), t] = 1  

	# calculate difference between predicted and true
	difference = softmax_probability - one_hot_matrix  
    # gradient is data term plus regularization term
	gradient = np.dot(x.T, difference) + alpha * w 

	gradient = gradient.flatten()
	return gradient

def gamma_logistic_regression_model(phi_train, phi_test, train_labels, test_labels, options, alpha=1e-6):
    ''' Train a logistic regression model on gamma dataset

		Arguments:
			phi_train: training dataset (NxM)
			phi_test: testing dataset (N'xM)
			train_labels: vector of training labels (Nx1)
			test_labels: vector of testing labels (N'x1)
            options: Dictionary object for scipy.optimize.minimize (dict())
            alpha: alpha value used in logistic loss and gradient functions (float)
            
            (N: number of elements in training data, N': number of elements in testing data)
		Returns:
			train_accuracy: (float)
		    train_loss: (float)
			test_accuracy: (float)
			test_loss: (float)
	'''
    
	# number of examples = N, number of features = M (for the NxM matrix)
    num_examples, num_features = phi_train.shape  
	# K = number of response classes
    K = len(np.unique(train_labels)) 
    # initialize weights
    weights = np.zeros(num_features * K)  
    
    # loss function
    def loss_function(w):
        return loss_func_logistic(w, phi_train, train_labels, K, alpha)
	# gradient function
    def grad_function(w):
        return gradient_loss_func_logistic(w, phi_train, train_labels, K, alpha)
    
    # optimize using scipy
    optimized = scipy.optimize.minimize(loss_function, weights, jac=grad_function, options=options)
    # reshape optimized weights
    weights_reshaped = optimized.x.reshape((num_features, K))

	# calculate training scores
    training_scores = np.dot(phi_train, weights_reshaped)  
	# calculate test scores
    test_scores = np.dot(phi_test, weights_reshaped) 

	# predict classes for training data
    train_predictions = np.argmax(training_scores, axis=1)  
	# predict classes for test data
    test_predictions = np.argmax(test_scores, axis=1)  

	# calculate training accuracy
    train_accuracy = np.mean(train_predictions == train_labels)  
	# calculate test accuracy
    test_accuracy = np.mean(test_predictions == test_labels)  

	# training loss
    train_loss = loss_func_logistic(weights_reshaped.flatten(), phi_train, train_labels, K, alpha) 
	# test loss
    test_loss = loss_func_logistic(weights_reshaped.flatten(), phi_test, test_labels, K, alpha)  

    return train_accuracy, train_loss, test_accuracy, test_loss

def toy_linear_regression_model(phi_train, phi_test, train_labels, test_labels):
    ''' Train a linear regression model on toy dataset

		Arguments:
			phi_train: training dataset with added bias term (NxM)
			phi_test: testing dataset with added bias term (N'xM)
			train_labels: matrix of one-hot encoded training labels (NxK)
			test_labels: matrix of one-hot encoded testing labels (N'xK)
               
            (N: number of elements in training data, N': number of elements in testing data)
		Returns:
			train_err: overall training error (float)
			test_err: overall testing error (float)
			w_hat: weights corresponding to the least squares prediction (MxK)
	'''
    print("\tTraining linear regression model...")
    
	# calculate least squares weights: W = (phi^T phi)^-1 phi^T T
    w_hat = np.linalg.pinv(np.dot(phi_train.T, phi_train))
    w_hat = np.dot(w_hat, np.dot(phi_train.T, train_labels))
    
    # predict class with highest score
    training_predictions = np.argmax(np.dot(phi_train, w_hat), axis=1)
    test_predictions = np.argmax(np.dot(phi_test, w_hat), axis=1)
    
    # calculate classification training error rate
    train_err = 1.0 - np.mean(training_predictions == np.argmax(train_labels, axis=1))
    # calculate classification test error rate
    test_err = 1.0 - np.mean(test_predictions == np.argmax(test_labels, axis=1))
    
    return train_err, test_err, w_hat  
	
def toy_logistic_regression_model(phi_train, phi_test, train_labels, test_labels, options, alpha=1e-6):
    ''' Train a linear regression model on toy dataset

		Arguments:
			phi_train: training dataset with added bias term (NxM)
			phi_test: testing dataset with added bias term (N'xM)
			train_labels: matrix of one-hot encoded training labels (NxK)
			test_labels: matrix of one-hot encoded testing labels (N'xK)
            options: Dictionary object for scipy.optimize.minimize (dict())
            alpha: alpha value used in logistic loss and gradient functions (float)

            (N: number of elements in training data, N': number of elements in testing data)
		Returns:
			train_err: overall training error (float)
			test_err: overall testing error (float)
			w_hat: weights corresponding to the logistic regression prediction (MxK)
	'''
    
    # number of examples = N, number of features = M (for the NxM matrix)
    num_examples, num_features = phi_train.shape  
	# K = number of response classes
    K = train_labels.shape[1]
	# initialize weights
    weights = np.zeros(num_features * K) 

	# turn one-hot to class indices for train set
    labels_train = np.argmax(train_labels, axis=1)  
    # turn one-hot to class indices for test set
    labels_test = np.argmax(test_labels, axis=1)

	# loss function
    def loss_function(w):
        return loss_func_logistic(w, phi_train, train_labels, K, alpha)
	# gradient function
    def grad_function(w):
        return gradient_loss_func_logistic(w, phi_train, train_labels, K, alpha)

    # optimize using scipy
    optimized = scipy.optimize.minimize(loss_function, weights, jac=grad_function, options=options)
    # reshape weights
    w_hat = optimized.x.reshape((num_features, K))  
    # predict classes for training data
    training_predictions = np.argmax(np.dot(phi_train, w_hat), axis=1)
    # predict classes for test data
    test_predictions = np.argmax(np.dot(phi_test, w_hat), axis=1)
    
	# calculate training error
    train_err = 1.0 - np.mean(training_predictions == labels_train)  
    # calculate test error
    test_err = 1.0 - np.mean(test_predictions == labels_test)  
    
    return train_err, test_err, w_hat 


### QUESTION 1

##### 1b

# define Pareto probability density function
def pareto(theta, alpha, beta):
    
    # create array of zeros with same shape as theta
    pdf = np.zeros_like(theta)
    # values where theta >= beta (Pareto defined)
    valid = theta >= beta
    # calculate PDF only where theta is valid
    pdf[valid] = alpha * beta**alpha / theta[valid]**(alpha + 1)
    return pdf

# theta values to plot over
theta_values = np.linspace(0.01, 10, 1000)

# (alpha, beta) pairs
pareto_parameters = [
    (0.1, 0.1),  
    (2.0, 0.1),  
    (1.0, 2.0)   
]

# plot
for alpha, beta in pareto_parameters:
    p_values = pareto(theta_values, alpha, beta)
    plt.plot(theta_values, p_values, label=f"alpha = {alpha}, beta = {beta}")

plt.title("Pareto Distributions (Question 1b)")
plt.xlabel("theta")
plt.ylabel("p(theta)")
plt.legend()
plt.grid(True)
plt.show()

##### 1f

# observations
x = np.array([0.7, 1.3, 1.7])
max_x = np.max(x)
N = len(x)

# (alpha, beta) pairs
aB_pairs = [(0.1, 0.1), (2.0, 0.1), (1.0, 2.0)]

# theta values to plot over
theta = np.linspace(0.01, 10, 1000)

# iterate through each alpha beta pair
for alpha, beta in aB_pairs:
    # posterior shape
    alpha_posterior_shape = alpha + N  
    # posterior scale
    beta_posterior_scale = max(beta, max_x)
    # create array of zeros with same shape as theta
    pdf = np.zeros_like(theta)  
    # values where theta >= beta (posterior defined)
    valid = theta >= beta_posterior_scale 
    # posterior density
    pdf[valid] = (alpha_posterior_shape * beta_posterior_scale**alpha_posterior_shape) / theta[valid]**(alpha_posterior_shape + 1)
    plt.plot(theta, pdf, label=f"alpha = {alpha}, beta = {beta}") 

# plot
plt.title("Posterior Distributions (Question 1f)")
plt.xlabel("theta")
plt.ylabel("p(theta | x)")
plt.legend()
plt.grid(True)
plt.show()


### QUESTION 2
def Question2():

    # Load the data
    data = np.load("gamma.npy", allow_pickle=True).item()
    x_train = data["train"]
    y_train = data["trainLabels"]
    x_test = data["test"]
    y_test = data["testLabels"]

	# Rescale all inputs for numerical stability
    x_offset = (np.min(x_train, 0) + np.max(x_train, 0)) / 2.0
    x_scale = (np.max(x_train, 0) - np.min(x_train, 0)) / 2.0
    x_train = (x_train - x_offset) / x_scale
    x_test = (x_test - x_offset) / x_scale

	# Set the regularization constant alpha=1e-6 and learn regression model
    ## using gtol instead of ftol as using ftol causes an error
    options = {"maxiter": 20000, "gtol": 1e-7}
    alpha = 1e-6

    print("Question 2")
    
	# feature sets to evaluate
    feature_sets = [(0, "Linear"), (1, "Diagonal Quadratic"), (2, "General Quadratic")]

	# iterate through each feature set
    for feature, name in feature_sets:
          # generate features for training
          phi_train = generate_features(x_train, feature)
          # generate features for testing
          phi_test = generate_features(x_test, feature)
          # train logistic regression, obtain training and testing performance
          train_acc, train_loss, test_acc, test_loss = gamma_logistic_regression_model(
          phi_train, phi_test, y_train, y_test, options, alpha)
          # print performance results
          print(f"\n{name} Features:")
          print(f"  Train Accuracy: {train_acc:.3f}, Train Neg Log-Post: {train_loss:.3f}")
          print(f"  Test Accuracy : {test_acc:.3f}, Test Neg Log-Post : {test_loss:.3f}")


### QUESTION 3
def Question3():

	# define toy datasets
    datasets = [
        ("partA_two_clouds.npy", "Question 3a"),
        ("partB_three_triangle.npy", "Question 3b"),
        ("partC_three_linear.npy", "Question 3c")
    ]

	# The basis function to use: Raw data and a constant (bias) feature
    def basis_function(x):
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=-1)

	# Set the regularization constant alpha=1e-6 and learn regression model
    ## using gtol instead of ftol as using ftol causes an error
    options = {"maxiter": 20000, "gtol": 1e-7}
    alpha = 1e-6

	# iterate through each toy dataset
    for dataset, name in datasets:
        print(f"\n{name}")
        # Load the data
        data = np.load(dataset, allow_pickle=True).item()
        x_train, x_test = data["Xtrain"], data["Xtest"]
        y_train, y_test = data["Ytrain"], data["Ytest"]

        # Rescale all inputs for numerical stability
        offset = (np.min(x_train) + np.max(x_train)) / 2.0
        scale = (np.max(x_train) - np.min(x_train)) / 2.0
        x_train = (x_train - offset) / scale
        x_test = (x_test - offset) / scale

		# compute linear features (add 1's column for intercept term)
        phi_train = basis_function(x_train)
        phi_test = basis_function(x_test)

        # train linear regression
        lin_train_err, lin_test_err, w_lin = toy_linear_regression_model(phi_train, phi_test, y_train, y_test)
        # calculate test accuracy
        lin_test_acc = 1.0 - lin_test_err

        # train logistic regression
        log_train_err, log_test_err, w_log = toy_logistic_regression_model(phi_train, phi_test, y_train, y_test, options, alpha)
        # calculate test accuracy
        log_test_acc = 1.0 - log_test_err

		# print accuracy results for linear and logistic regression
        print(f"Linear Regression Test Accuracy : {lin_test_acc:.3f}")
        print(f"Logistic Regression Test Accuracy : {log_test_acc:.3f}")

        # plot
        plotter_classifier(w_lin, basis_function, x_train, np.argmax(y_train, axis=1), title="Linear Regression")
        plt.show()
        plotter_classifier(w_log, basis_function, x_train, np.argmax(y_train, axis=1), title="Logistic Regression")
        plt.show()


if __name__ == "__main__":
    Question2()
    Question3()
    # example functions:
    # example_of_quadratic_use()
    
    #break to observe output
    # input("Press Enter to continue to toy dataset example...")
    
    # example_of_plot_function_with_toy_dataset()
