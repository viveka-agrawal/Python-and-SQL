# Standard Imports
import numpy as np
import matplotlib.pyplot as plt

# Imports to use for optimization
from functools import partial
import scipy.optimize

def plotter_classifier(w,basis_func, x, y, title=None, axis=None, grid_density=1000):
	'''
		Adopted from the mltools packaged created by Alex Ihler.

		Plots a 2D linear classifier along with the decision boundaries
		
		Parameters:
			w: The weights of a linear classifier
			basis_func: A python function that can create the basis function for the data points
			x: data to plot
			y: Classes of x
			title: A title to give the plot
			axis: a matplotlib axis.  If one is not provided then "plt" will be used
			grid_density: How dense of a grid should be used for rendering the decision boundary
	'''
	# Makes sure the data is only 2D
	if (x.shape[1] != 2):
		raise ValueError('plotter_classifier: function can only be called using two-dimensional data (features)')

	# Define an axis if one isnt passed in
	if(axis == None): 
		axis = plt 
	# axis.hold(True)
	axis.plot(x[:,0],x[:,1], color="black", visible=False )

	# TODO: can probably replace with final dot plot and use transparency for image (?)
	ax = axis.axis()
	xticks = np.linspace(ax[0],ax[1],grid_density)
	yticks = np.linspace(ax[2],ax[3],grid_density)
	grid = np.meshgrid( xticks, yticks )
	x_grid = np.column_stack( (grid[0].flatten(), grid[1].flatten()) )

	# Plot the colors
	grid_phi = basis_func(x_grid)
	y_hat_grid = np.argmax(np.matmul(grid_phi, w), 1)
	axis.imshow( y_hat_grid.reshape( (len(xticks),len(yticks)) ), extent=ax, interpolation='nearest',origin='lower',alpha=0.5, aspect='auto' )

	cmap = plt.cm.get_cmap()

	classes = np.unique(y)

	cvals = (classes - np.min(classes)) / (np.max(classes)-np.min(classes)+1e-100)
	for i,c in enumerate(classes): 
		axis.scatter( x[y==c,0],x[y==c,1], edgecolors="black", color=cmap(cvals[i]))  
	axis.axis(ax)

	if(title is not None):
		axis.title(title)


def generate_features(x, feature_type):
	'''Function to generate feature mappings on the gamma dataset.
	
		Arguments:
			x: data features (either training or testing data) (NxD)
			feature_type: index determining which of the three feature types to generate 
							(int={0:linear, 1:diagonal quadratic, 2:general quadratic})
		Returns:
			phi: New feature data (either: Nx[D+1], Nx[2D+1], or Nx[(D+1)D/2+D+1])
	'''
	num_features = x.shape[-1]

	if(feature_type == 0):
		# Constant bias plus linear features
		phi = np.zeros((x.shape[0], num_features+1))
		phi[:, 0] = 1.0
		phi[:, 1:] = x

	elif(feature_type == 1):
		# Constant bias plus linear plus squares of individual features
		phi = np.zeros((x.shape[0], (num_features*2)+1))
		phi[:, 0] = 1.0
		phi[:, 1:(num_features+1)] = x
		phi[:, (num_features+1):] = x**2

	elif(feature_type == 2):
		# General quadratic with products of all pairs of feature values

		num_phi = int(((num_features+1)*num_features/2) + num_features + 1)
		phi = np.zeros((x.shape[0], num_phi))
		phi[:, 0] = 1.0
		phi[:, 1:(num_features+1)] = x
		phi[:, (num_features+1):((2*num_features)+1)] = x**2

		# This isnt the most efficient way of doing this but it gets the 
		# job done for this small dataset
		for i in range(x.shape[0]):
			x_feats = x[i]
			curr_idx = (2*num_features) +1
			for j in range(x_feats.shape[0]):
				for k in range(x_feats.shape[0]):
					if(k <= j):
						continue
					phi[i, curr_idx] = x_feats[j] * x_feats[k]
					curr_idx += 1
	else:
		# Dont know what this type is
		assert(False)		

	return phi

#========================================================================================
# Quadratic loss functions (gives examples of syntax, NOT required for solutions)
#========================================================================================

def loss_func_quadratic(w, x, t, alpha):
	''' Compute a quadratic loss function for linear prediction of classification targets

		Arguments:
			w: weights
			x: data features
			t: data targets
			alpha: regularization constant

		Returns:
			The loss function value
	'''
	delta = t - np.matmul(x, np.expand_dims(w, 1)).squeeze()
	loss = (0.5*alpha*np.sum(w**2)) + (0.5*np.sum(delta**2))
	return loss

def gradient_loss_func_quadratic(w, x, t, alpha):
	''' Compute the gradient (with respect to w) of the quadratic classification loss

		Arguments:
			w: weights
			x: data features
			t: data targets
			alpha: regularization constant

		Returns:
			The gradient of loss function with respect to w
	'''
	delta = t - np.matmul(x, np.expand_dims(w, 1)).squeeze()
	gradient = (alpha*w) - np.matmul(np.transpose(x),np.expand_dims(delta,1).squeeze())
	return gradient

#=============================================================================================
# example use case functions (gives examples of use case syntax, NOT required for solutions)
#=============================================================================================
def example_of_quadratic_use():
	'''
	Example of quadratic function use
	'''
	print('loading data:')
	# Load the data
	data = np.load("gamma.npy", allow_pickle=True).item()
	train = data["train"]
	train_labels = data["trainLabels"]
	test = data["test"]
	test_labels = data["testLabels"]

	# Rescale all inputs for numerical stability
	x_offset = (np.min(train, 0) + np.max(train, 0)) / 2.0
	x_scale  = (np.max(train, 0) - np.min(train, 0)) / 2.0
	train = (train - x_offset) / x_scale
	test  = (test - x_offset) / x_scale

	# Below is an example of doing "linear regression for classification",
	# using a regularized quadratic loss function we have provided.
	# You need to instead implement and test a multinomial logistic loss function.

	# Set the regularization constant alpha=1e-6 and learn regression model
	alpha = 1e-6
	M = train.shape[1]
	w0 = np.zeros((M,))

	# Create the partial functions that we will be using for the optimization.
	# These functions bind the loss and gradient functions such that we can just 
	# pass in a value for w and evaluate the function.
	# You must derive and implement the gradient function, NOT approximate numerically.
	loss_func = partial(loss_func_quadratic, x=train, t=train_labels, alpha=alpha)
	grad_func = partial(gradient_loss_func_quadratic, x=train, t=train_labels, alpha=alpha)

	# Optimize using the scipy optimization library
	# Here we use the BFGS optimization method which takes into account the history
	# of gradient evaluations during optimization to accelerate convergence.
	print('Training model with quadratic loss:')
	options = dict()
	options["maxiter"] = 20000
	options["ftol"] = 1e-7
	results = scipy.optimize.minimize(fun=loss_func, x0=w0, jac=grad_func, method="L-BFGS-B", options=options)
	assert(results.success) # make sure it converged
	w_min = results.x  # the value for w that minimizes the loss func

	# Compute the value of the minimized loss
	train_loss = loss_func(w_min)

	# Evaluate accuracy on test data
	t_hat = np.matmul(test, np.expand_dims(w_min, 1)).squeeze() > 0.5
	test_accuracy = np.sum(t_hat == test_labels)/float(test_labels.shape[0])

	print("  Train Loss: {:0.3g}".format(train_loss))
	print("  Test Accuracy: {:0.3f}".format(test_accuracy))

def example_of_plot_function_with_toy_dataset():
	'''
	Plot function example using toy dataset
	'''
	print('loading data:')
	# Load data: uncomment appropriate filename
	data = np.load("partA_two_clouds.npy", allow_pickle=True).item()
	# data = np.load("partB_three_triangle.npy", allow_pickle=True).item()
	# data = np.load("partC_three_linear.npy", allow_pickle=True).item()

	# Unpack the data
	x_train = data["Xtrain"]
	x_test = data["Xtest"]
	y_train = data["Ytrain"]
	y_test = data["Ytest"]

	# Properties of dataset
	num_train = x_train.shape[0]
	num_test  = x_test.shape[0]
	num_class = y_train.shape[1]

	# Rescale all inputs so training lies in [-1,+1] for numerical stability
	x_offset = (np.min(x_train) + np.max(x_train)) / 2.0 
	x_scale  = (np.max(x_train) - np.min(x_train)) / 2.0
	x_train_rescaled = (x_train - x_offset) / x_scale
	x_test_rescaled  = (x_test - x_offset) / x_scale

    # The basis function to use: Raw data and a constant (bias) feature
	def basis_function(x):
		return np.concatenate([np.ones((x.shape[0],1)), x], axis=-1)

	# compute linear features (add 1's column for intercept term)
	phi_train = basis_function(x_train_rescaled)
	phi_test =  basis_function(x_test_rescaled)

	# Example: compute and plot accuracy for random classification boundaries
	w_hat = np.concatenate([np.zeros((1, num_class)), np.random.normal(0, 1, (2, num_class))], axis=0)

	f_hat_test = np.matmul(phi_test, w_hat)
	y_hat_test = np.argmax(f_hat_test,1)
	y_int_test = np.argmax(y_test,1)
	test_accuracy = np.sum( y_hat_test == y_int_test ) / num_test
	print("Accuracy on Test : {:0.3f}".format(test_accuracy))

	# Plot!
	plotter_classifier(w_hat, basis_function, x_test_rescaled, y_int_test, title="Linear Regression")
	plt.show()

	

	
