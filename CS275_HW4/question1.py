import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, acquisition
from bayesian_optimization_utils import *
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# Settings for Bayesian Optimization runs
num_runs_per_experiment = 5
num_bo_steps = 50
plotter = Plotter()  # Initialize the plotter

# -----------------------
# 1a) UCB Acquisition Function
# -----------------------
plotter.reset()  # Reset plotter for a new set of plots
kappa_values = [1.0, 5.0, 10.0]

for kappa in kappa_values:
    all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
    for exp_run_iter in range(num_runs_per_experiment):
        # Create the BO optimizer with UCB acquisition
        acquisition_function = acquisition.UpperConfidenceBound(kappa=kappa)
        bayesian_optimizer = BayesianOptimization(
            negative_modified_branin_func,
            get_negative_modified_branin_func_bounds(),
            verbose=0,
            acquisition_function=acquisition_function
        )

        # Initialize with 2 random points
        bayesian_optimizer.maximize(init_points=2, n_iter=0)

        # Run Bayesian optimization for num_bo_steps
        for bo_iter in range(num_bo_steps):
            bayesian_optimizer.maximize(init_points=0, n_iter=1)  # Single BO step
            # Store the maximum function value found so far
            best_x1 = bayesian_optimizer.max["params"]["x1"]
            best_x2 = bayesian_optimizer.max["params"]["x2"]
            all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

    # Add the results for this kappa value to the plotter
    plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, f"UCB (kappa={kappa})")

# Create and display the plot for UCB
fig_ucb = plotter.create_figure(1)
plt.figure(fig_ucb.number)  # Ensure the figure is active for display
plt.show()

# -----------------------
# 1b) POI Acquisition Function
# -----------------------
plotter.reset()
xi_values = [0.01, 0.1, 1.0]

for xi in xi_values:
    all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
    for exp_run_iter in range(num_runs_per_experiment):
        acquisition_function = acquisition.ProbabilityOfImprovement(xi=xi)
        bayesian_optimizer = BayesianOptimization(
            negative_modified_branin_func,
            get_negative_modified_branin_func_bounds(),
            verbose=0,
            acquisition_function=acquisition_function
        )

        bayesian_optimizer.maximize(init_points=2, n_iter=0)

        for bo_iter in range(num_bo_steps):
            bayesian_optimizer.maximize(init_points=0, n_iter=1)
            best_x1 = bayesian_optimizer.max["params"]["x1"]
            best_x2 = bayesian_optimizer.max["params"]["x2"]
            all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

    plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, f"POI (xi={xi})")

fig_poi = plotter.create_figure(2)
plt.figure(fig_poi.number)
plt.show()

# -----------------------
# 1c) EI Acquisition Function
# -----------------------
plotter.reset()
xi_values = [0.01, 0.1, 1.0]

for xi in xi_values:
    all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
    for exp_run_iter in range(num_runs_per_experiment):
        acquisition_function = acquisition.ExpectedImprovement(xi=xi)
        bayesian_optimizer = BayesianOptimization(
            negative_modified_branin_func,
            get_negative_modified_branin_func_bounds(),
            verbose=0,
            acquisition_function=acquisition_function
        )

        bayesian_optimizer.maximize(init_points=2, n_iter=0)

        for bo_iter in range(num_bo_steps):
            bayesian_optimizer.maximize(init_points=0, n_iter=1)
            best_x1 = bayesian_optimizer.max["params"]["x1"]
            best_x2 = bayesian_optimizer.max["params"]["x2"]
            all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

    plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, f"EI (xi={xi})")

fig_ei = plotter.create_figure(3)
plt.figure(fig_ei.number)
plt.show()

# -----------------------
# 1d) Matern Kernel Parameter
# -----------------------
plotter.reset()
xi = 0.01  # Fixed EI xi
nu_values = [0.5, 2.5, 10.0]

for nu in nu_values:
    all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
    for exp_run_iter in range(num_runs_per_experiment):
        acquisition_function = acquisition.ExpectedImprovement(xi=xi)
        bayesian_optimizer = BayesianOptimization(
            negative_modified_branin_func,
            get_negative_modified_branin_func_bounds(),
            verbose=0,
            acquisition_function=acquisition_function
        )

        # Change the kernel
        kernel = Matern(nu=nu)
        bayesian_optimizer._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5,
            random_state=bayesian_optimizer._random_state
        )

        bayesian_optimizer.maximize(init_points=2, n_iter=0)

        for bo_iter in range(num_bo_steps):
            bayesian_optimizer.maximize(init_points=0, n_iter=1)
            best_x1 = bayesian_optimizer.max["params"]["x1"]
            best_x2 = bayesian_optimizer.max["params"]["x2"]
            all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

    plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, f"EI (xi={xi}), Matern (nu={nu})")

fig_matern = plotter.create_figure(4)
plt.figure(fig_matern.number)
plt.show()

# -----------------------
# 1e) RBF Kernel Lengthscale
# -----------------------
plotter.reset()
xi = 0.01  # Fixed EI xi
sigma_values = [0.001, 1.0, 3.0]

for sigma in sigma_values:
    all_runs_estimated_max_value_per_iter = np.zeros((num_runs_per_experiment, num_bo_steps))
    for exp_run_iter in range(num_runs_per_experiment):
        acquisition_function = acquisition.ExpectedImprovement(xi=xi)
        bayesian_optimizer = BayesianOptimization(
            negative_modified_branin_func,
            get_negative_modified_branin_func_bounds(),
            verbose=0,
            acquisition_function=acquisition_function
        )

        # Change the kernel
        kernel = RBF(length_scale=sigma)
        bayesian_optimizer._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=bayesian_optimizer._random_state
        )

        bayesian_optimizer.maximize(init_points=2, n_iter=0)

        for bo_iter in range(num_bo_steps):
            bayesian_optimizer.maximize(init_points=0, n_iter=1)
            best_x1 = bayesian_optimizer.max["params"]["x1"]
            best_x2 = bayesian_optimizer.max["params"]["x2"]
            all_runs_estimated_max_value_per_iter[exp_run_iter, bo_iter] = negative_modified_branin_func(best_x1, best_x2)

    plotter.add_experiment_plot(all_runs_estimated_max_value_per_iter, f"EI (xi={xi}), RBF (sigma={sigma})")

fig_rbf = plotter.create_figure(5)
plt.figure(fig_rbf.number)
plt.show()