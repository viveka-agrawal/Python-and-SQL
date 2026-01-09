
# Some imports we will need
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def area_roc(confidence, test_class, do_plot=False, num_points=500, color="red",linewidth=2.0):
	''' Compute and plot ROC curve and corresponding area
	
		Compute ROC curve, by considering a range of thresholds of confidence values,
		and computing empirical false alarm and detection rates for each one.
		For probabilistic models, these confidence values are normally either a
		log-likelihood ratio between the two classes, or the Bayesian posterior
		probability of class label 1.

		Parameters:
			confidence: (N, ) array of scores for each test example, where higher scores indicate greater confidence in target presence
			test_class: (N, ) array of giving ground truth for each test example, where 1 indicates target presence and 0 target absence
			do_plot: Boolean.  Will plot if true
			num_points: number of false alarm rates at which to evaluate curve 
			color: The color of the line for the plot
			linewidth: The line width of the line for the plot

		Returns: 
			area_roc, alarm_rate, detect_rate

			area_roc : area under ROC curve
			alarm_rate: false alarm rates, uniformly sampled on [0,1]
			detect_rate = corresponding detection rates computed from given confidence
	'''

	# perturb confidence to avoid "staircase" effect
	confidence += np.random.uniform(low=0.0, high=1.0, size=confidence.shape)*1e-10

	# indices of negative and positive test cases
	index_absent = (test_class <= 0)
	index_present = (test_class >= 1)

	# Extract the confidences values for the absent and present cases
	confidence_for_absent = confidence[index_absent]
	confidence_for_present = confidence[index_present]

	# Sort the confidence values in ascending order
	confidence_for_absent = np.sort(confidence_for_absent) 
	confidence_for_present = np.sort(confidence_for_present)

	# compute ROC curve
	resamp_indices = np.linspace(0, confidence_for_absent.shape[0]-1e-8, num_points, endpoint=True)
	resamp_indices = np.fix(resamp_indices).astype("int")
	confidence_for_absent_resampled = confidence_for_absent[resamp_indices]

	# Compute the alarm and detect rates
	alarm_rate = np.zeros(shape=(num_points,))
	detect_rate = np.zeros(shape=(num_points,))
	for i in range(num_points):
		detect_rate[i] = np.sum(confidence_for_present >= confidence_for_absent_resampled[i]) / np.sum(index_present)
		alarm_rate[i]  = np.sum(confidence_for_absent  >= confidence_for_absent_resampled[i]) / np.sum(index_absent)

	# Compute area under ROC curve 
	area_roc = ((alarm_rate[1:]-alarm_rate[:-1]) * (detect_rate[1:]+detect_rate[:-1])) / 2.0
	area_roc = np.abs(np.sum(area_roc))

	# If we should plot then plot
	if(do_plot):
		plt.plot(alarm_rate, detect_rate, color=color, linewidth=linewidth)
		plt.ylabel("Detection Rate");
		plt.xlabel("False Alarm Rate");
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)
		plt.show()

	return area_roc, alarm_rate, detect_rate