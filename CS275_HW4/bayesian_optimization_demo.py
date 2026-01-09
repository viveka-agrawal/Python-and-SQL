# UCI CS275P: Bayesian Optimization via Gaussian Processes
# This is DEMONSTRATION code:
# - It gives examples of how to load the Bayesian Optimization code,
#   and call it with different kernel and acquisition functions
# - It defines and plots the 2D Branin-Hoo objective function 
# - You may (but do not have to) reuse parts of this code in your solutions. 
# - It is NOT a template for the individual questions you must answer,
#   for that see the main homework pdf.

# Our standard numpy and matplotlib imports
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
# Also need to include a Bayesian optimization package. 
# To install:
#      pip install bayesian-optimization
#          or
#      https://anaconda.org/conda-forge/bayesian-optimization

from bayes_opt import BayesianOptimization, acquisition

# Imports for the kernels and GP that the BO package uses
from sklearn.gaussian_process.kernels import Matern, RBF 
from sklearn.gaussian_process import GaussianProcessRegressor

# Homework Imports
from bayesian_optimization_utils import *

#################################
## Plot the Branin Function in 3D
#################################

density = 30
x1 = np.linspace(-5, 10, density)
x2 = np.linspace(0, 15, density)
X1, X2 = np.meshgrid(x1, x2)

Z = negative_modified_branin_func(X1, X2)
fig = plt.figure(0)
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)');

################################################
## Settings for all Bayesian Optimization trials
################################################

# The number of times we should run each experiment
num_runs_per_experiment = 5

# The number of BO steps to take for each experiment
num_bo_steps = 50

# The plotter class that we will use for plotting everything
plotter = Plotter()

#######################################################
# Baseline:  Choose function evaluation points randomly
#######################################################
all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
for exp_run_iter in range(num_runs_per_experiment):

	# Create the BO optimizer
	bayesian_optimizer = BayesianOptimization(negative_modified_branin_func, get_negative_modified_branin_func_bounds(), verbose=0)

	# !!! YOU ALWAYS HAVE TO DO THIS FOR THIS BO PACKAGE TO WORK !!!
	# Initialize with 2 random points!  (No need to set an acquisition function)
	bayesian_optimizer.maximize(init_points=2, n_iter=0)

	# Do many iterations 1 at a time so we can record the current best guess for the max
	for bo_iter in range(num_bo_steps):

		# Draw random samples
		bayesian_optimizer.maximize(init_points=1, n_iter=0)

		# Grab the current max after the iteration
		best_x1 = bayesian_optimizer.max["params"]["x1"]
		best_x2 = bayesian_optimizer.max["params"]["x2"]
		all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, "Random Samples")

#####################################################
# Bayesian Optimization with UCB acquisition function
#####################################################
all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
for exp_run_iter in range(num_runs_per_experiment):

    # Create the BO optimizer: UCB with kappa=3.0
	acquisition_function = acquisition.UpperConfidenceBound(kappa=3.0)
	bayesian_optimizer = BayesianOptimization(negative_modified_branin_func, \
        get_negative_modified_branin_func_bounds(), verbose=0, \
        acquisition_function=acquisition_function)

	# !!! YOU ALWAYS HAVE TO DO THIS FOR THIS BO PACKAGE TO WORK !!!
	# Initialize with 2 random points!  (No need to set an acquisition function)
	bayesian_optimizer.maximize(init_points=2, n_iter=0)

	# Do many iterations 1 at a time so we can record the current best guess for the max
	for bo_iter in range(num_bo_steps):

		# Do 1 step BO optimization
		bayesian_optimizer.maximize(init_points=0, n_iter=1)

		# Grab the current max after the iteration
		best_x1 = bayesian_optimizer.max["params"]["x1"]
		best_x2 = bayesian_optimizer.max["params"]["x2"]
		all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, "UCB")


############################################
## Change Kernel Function from Matern to RBF
############################################
all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
for exp_run_iter in range(num_runs_per_experiment):

    # Create the BO optimizer: UCB with kappa=3.0
	acquisition_function = acquisition.UpperConfidenceBound(kappa=3.0)
	bayesian_optimizer = BayesianOptimization(negative_modified_branin_func, \
        get_negative_modified_branin_func_bounds(), verbose=0, \
        acquisition_function=acquisition_function)

	# Change the kernel from default (Matern(nu=2.5)) to RBF
	kernel = RBF(length_scale=1.0)
	bayesian_optimizer._gp =  GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True, n_restarts_optimizer=5, random_state=bayesian_optimizer._random_state)

	# !!! YOU ALWAYS HAVE TO DO THIS FOR THIS BO PACKAGE TO WORK !!!
	# Initialize with 2 random points!  (No need to set an acquisition function)
	bayesian_optimizer.maximize(init_points=2, n_iter=0)

	# Do many iterations 1 at a time so we can record the current best guess for the max
	for bo_iter in range(num_bo_steps):

		# Do 1 step BO optimization
		bayesian_optimizer.maximize(init_points=0, n_iter=1)

		# Grab the current max after the iteration
		best_x1 = bayesian_optimizer.max["params"]["x1"]
		best_x2 = bayesian_optimizer.max["params"]["x2"]
		all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

# Add the data to the plotter
plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, "RBF Kernel")

# Create the figure
fig1 = plotter.create_figure(1)

plt.show()


