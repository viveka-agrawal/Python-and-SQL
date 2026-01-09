# Our standard numpy and matplotlib imports
import numpy as np
import matplotlib.pyplot as plt

def negative_modified_branin_func(x1, x2):
	'''
		Implement the modified Branin-Hoo function where 5x_1 is added to the
		standard Branin-Hoo function.

		Note: this returns the negative of the modified Branin function 
				since our BO package can only fund the max and not the min
				so to find the min we maximize the negative of the function 
		
		Global Max of this (negative) function:
			f(x1, x2) global min: 16.64402157084318897540298167
				at
			(x1, x2)≈(-3.689285272561052366675585548, 13.629987728944896052816531259)
		solved using Wolfram Alpha: https://www.wolframalpha.com/input?i=global+max+of+2d+branin+func&assumption=%7B%22F%22%2C+%22GlobalMaximizeCalculator%22%2C+%22curvefunction%22%7D+-%3E%22-%28%28y+-+%285.1%2F%284pi%5E2%29%29x%5E2+%2B%285%2Fpi%29x+-+6%29%5E2+%2B+10%281-%281%2F%288pi%29%29%29cos%28x%29+%2B+10+%2B5x%29%22&assumption=%7B%22F%22%2C+%22GlobalMaximizeCalculator%22%2C+%22domain%22%7D+-%3E%22-5%3C%3Dx%3C%3D10+and+0%3C%3Dy%3C%3D15%22

		References:
			-https://www.sfu.ca/~ssurjano/branin.html

		Parameters:
			x1: numpy array of inputs for x1
			x2: numpy array of inputs for x2 with the same shape as x1

		Return:
			Negative Modified Branin Function
	'''
	a = 1.0
	b = 5.1 / (4.0*(np.pi**2))
	c = 5.0 / np.pi
	r = 6.0
	s = 10.0
	t = 1.0 / (8*np.pi)

	# Standard Branin-Hoo (or just Branin) function 
	term1 = a* ((x2 - (b*(x1**2)) + (c*x1) - r)**2)
	term2 = s*(1-t)*np.cos(x1)
	y = term1 + term2 + s

	# Modified Branin func
	y += (5.0*x1)

	# Negative modified Branin func
	y = -y

	return y


def get_negative_modified_branin_func_bounds():
	'''
		Get the bounds for the (negative) Modified Branin Function

		Returns:
			{"x1": (-5, 10), "x2": (0, 15)}
	'''

	return {"x1": (-5, 10), "x2": (0, 15)}


class Plotter:
	def __init__(self):

		self.data_means = []
		self.data_range_upper = []
		self.data_range_lower = []
		self.data_label = []

	def add_experiment_plot(self, all_experiment_runs, label):

		self.data_label.append(label)

		# Compute the mean of the estimates
		self.data_means.append(np.mean(all_experiment_runs, axis=0))

		# Compute the lower and upper range data
		self.data_range_upper.append(np.max(all_experiment_runs, axis=0))
		self.data_range_lower.append(np.min(all_experiment_runs, axis=0))

	def create_figure(self, fig_num=1):

		# Figure out how many subplots we need
		number_subplots = len(self.data_range_upper)

		cols = int(np.sqrt(number_subplots))
		rows = cols
		while((cols*rows < number_subplots)):
			rows += 1

		# Make the figure
		# fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(10, 12))
		fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False)

		# Hack for if we have 1 plot....
		if(number_subplots == 1):
			axes = [axes]
		else:	
			axes = axes.reshape(-1,)

		for i in range(number_subplots):
			ax = axes[i]

			# Plot the true Max
			ax.axhline(y=16.64402157084318897540298167, label="True Max (16.644)", color="red")

			# Unpack the curve data
			mean_max_value = self.data_means[i]
			range_upper = self.data_range_upper[i]
			range_lower = self.data_range_lower[i]
			label  = self.data_label[i]
			
			# Create the error bar uncertainty
			yerr = np.zeros((2, range_upper.shape[0]))
			yerr[1,:] = range_upper - mean_max_value
			yerr[0,:] = mean_max_value - range_lower

			# Plot, make x start from 1 since this is after the first iteration
			iters = np.arange(1, mean_max_value.shape[0]+1)
			(_, caps, _) = ax.errorbar(iters, mean_max_value, yerr=yerr, label=label, capthick=1, fmt="-o", markersize=3, elinewidth=1, capsize=6)

			# Make it look pretty
			ax.legend(loc="lower right")
			ax.set_xlabel("BO Steps")
			ax.set_ylabel("Max Value")
			ax.set_ylim([np.min(range_lower)-5, 20])

		return fig