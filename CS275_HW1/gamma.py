# UCI CS275P: Gaussian Naive Bayes Classification of Gamma Telescope Data

# Some imports we will need
import numpy as np

# Load the gamma dataset 
# train: training matrix of 10 continuous observations from star showers.
# train_labels: column vector of {0,1} training labels where 0=hadron,1=gamma.
# test: test matrix of 10 continuous observations from star showers
# test_labels: column vector of {0,1} test labels where 0=hadron,1=gamma.
data = np.load("gamma.npy", allow_pickle=True).item()
train = data["train"]
train_labels = data["trainLabels"]
test = data["test"]
test_labels = data["testLabels"]

def calculate_mean_and_var(train, train_labels):
	''' 
	Function to calculate the means and variances of a dataset.
    
    Parameters
    ----------
    train :
        train data (loaded from data on line 15)
	train_labels :
        train labels (loaded from data on line 16)
    Returns
    -------
    p_y :
        prior class probabilities P(Y) (shape should be: 2x1)
	mu_hat :
		means of each class in train (shape should be: 2xD)
	sigma_hat : 
		variances of each class in train (shape should be: 2xD) 
    '''

	# Estimate P(Y) (class prior probs)

	# prior probs P(Y=0) & P(Y=1)
	p_y = np.zeros(shape=(2,))
	# calculate P(Y=0)
	p_y[0] = np.mean(train_labels == 0)
	# calculate P(Y=1)
	p_y[1] = np.mean(train_labels == 1)


	# Estimate P(X_j|Y) (class conditional stats)

	# number of features in input data
	num_dims = train.shape[1] 
	# initialize mean estimates of each class (2 X num features)
	mu_hat = np.zeros(shape=(2,num_dims))
	# initialize variance estimates of each class (2 X num features)
	sigma_hat = np.zeros(shape=(2,num_dims)) 


	# MLE estimates for mu and sigma

	# iterate through class 0 and class 1
	for i in range(2): 
		# select data points in current class (either 0 or 1)
		class_samples = train[train_labels == i] 
		# calculate mean for each feature in current class
		mu_hat[i, :] = np.mean(class_samples, axis=0) 
		# calculate variance for each feature in current class
		sigma_hat[i, :] = np.var(class_samples, axis=0) 

	# return prior probs, means, and variances
	return p_y, mu_hat, sigma_hat

def calculate_log_probability(p_y, mu_hat, sigma_hat, test):
	''' 
	Function to calculate the joint log-probability of each test example.
    
    Parameters
    ----------
    p_y :
        prior class probabilities from calculate_mean_and_var
	mu_hat :
        means of each class from calculate_mean_and_var
	sigma_hat:
		variances of each class from calculate_mean_and_var
	test:
        test data (loaded from data on line 17, shape should be: MxD)
    Returns
    -------
    log_p_y :
        joint log-probability log P(Y,X) = log P(Y) + log P(X|Y)
        of each test example, given each possible class label (shape: Mx2)
        log_p_y[i,0] = log P(Y=0,X=test[i,:])
        log_p_y[i,1] = log P(Y=1,X=test[i,:])

    '''
	# calculate P(Y|X)

	# matrix to store joint log-probs for each class
	log_p_y = np.zeros(shape=(test.shape[0], 2))

	# number of classes (2 because there is class 0 and class 1)
	num_classes = p_y.shape[0]                   


	# calculate P(X|Y) using Gaussian Naive Bayes formula

	# iterate through each class
	for classes in range(num_classes): 
		# calculate -1/2 * sum of (log(2*pi*variance)) which is log term of 1st term in GNB formula
		# first term uses pi and sigma squared (which is variance)
		pi_sigma = -0.5 * np.sum(np.log(2 * np.pi * sigma_hat[classes, :]))  
		# calculate -1/2 * sum of ((feature - mean)^2 / variance) which is log term of 2nd term in GNB formula
		# second term uses x (test data), mu (mean), and sigma squared (variance)
		x_mu_sigma = -0.5 * np.sum(((test - mu_hat[classes, :]) ** 2) / sigma_hat[classes, :], axis=1)  
		# add both log terms to get log-likelihood of X given Y
		log_likelihood = pi_sigma + x_mu_sigma  
        # add log prior P(Y) to log-likelihood to get joint log-prob logP(Y,X)
		log_p_y[:, classes] = log_likelihood + np.log(p_y[classes])
	
	# return matrix of joint log-probs
	return log_p_y

def calculate_map_tpr_and_fpr(log_p_y, test_labels):
	''' 
	Function to calculate the true-positive rate and false-positive rate,
    for the MAP Bayesian classifier that minimizes the probability of error.
    
    Parameters
    ----------
    log_p_y :
        joint log-probabilities of test data, from calculate_log_probability.
	test_labels :
        test labels (should be loaded from data on line 18).
    Returns
    -------
	tpr :
		True positive rate (float value).
	fpr : 
		False positive rate (float value).    
    '''
	## MAP Estimate minimizes 0-1 Loss

	# predict class with highest joint log-prob
	y_hat = np.argmax(log_p_y, axis=1)
	# count number of true positive samples
	num_pos = np.sum(test_labels==1)
	# count number of true negative samples
	num_neg = np.sum(test_labels==0)

	# predicted class 1, actual class 1
	true_pos = np.sum((y_hat == 1) & (test_labels == 1))
	# predicted class 1, actual class 0
	false_pos = np.sum((y_hat == 1) & (test_labels == 0))

	# true positive rate: fraction of actual positives correctly predicted
	tpr = true_pos / num_pos
	# false positive rate: fraction of actual negatives incorrectly predicted
	fpr = false_pos / num_neg

	# return true positive rate and negative true rate 
	return tpr, fpr

def calculate_costly_tpr_and_fpr(log_p_y, test_labels, background_cost):
	''' 
	Function to calculate the true-positive rate and false-positive rate,
    for a Bayesian classifier that minimizes a weighted classification loss.
    
    Parameters
    ----------
    log_p_y :
        joint log-probabilities of test data, from calculate_log_probability.
	test_labels :
        test labels (should be loaded from data on line 18).
    background_cost :
        cost factor for classifying positive examples as negative (missed detections),
        relative to cost of classifying negative examples as positive (false alarms);
        this scalar is 50 for the scenario in part (e) of the HW handout.
    Returns
    -------
	tpr :
		True positive rate (float value).
	fpr : 
		False positive rate (float value).    
    '''

	# calculate log posterior odds (posterior probability ratio)
	# log P(Y=1,X) - log P(Y=0,X)
	post_pr_y = log_p_y[:, 1] - log_p_y[:, 0]

	# minimize expected loss
	# calculate log-threshold using misclassification cost and log priors
	cost_prior_ratio = np.log(background_cost) + np.log(p_y[0]) - np.log(p_y[1])


	## COSTLY False negatives

	# count number of actual positives and negatives
	num_pos = np.sum(test_labels==1)
	num_neg = np.sum(test_labels==0)
	# assign class 1 if log-ratio > threshold
	y_hat = (post_pr_y > cost_prior_ratio).astype(int)

	# predicted class 1, actual class 1
	true_pos = np.sum((y_hat == 1) & (test_labels == 1))
	# predicted class 1, actual class 0
	false_pos = np.sum((y_hat == 1) & (test_labels == 0))

	# true positive rate: fraction of actual positives correctly predicted
	if num_pos > 0:
		tpr = true_pos / num_pos
	else:
		# float value
		tpr = 0.0
		
	# false positive rate: fraction of actual negatives incorrectly predicted
	if num_neg > 0:
		fpr = false_pos / num_neg
	else:
		# float value
		fpr = 0.0

	# return true positive rate and negative true rate
	return tpr, fpr

# estimate parameters from training data
p_y, mu_hat, sigma_hat = calculate_mean_and_var(train, train_labels)
# calculate log-probs of test samples for each class
log_p_y = calculate_log_probability(p_y, mu_hat, sigma_hat, test)

# Question 3d: 
tpr_map, fpr_map = calculate_map_tpr_and_fpr(log_p_y, test_labels)
print(f"Question 3d:")
print(f"True Positive Rate = {tpr_map:.4f}")
print(f"False Positive Rate = {fpr_map:.4f}")

# Question 3e: 
background_cost = 50
tpr_costly, fpr_costly = calculate_costly_tpr_and_fpr(log_p_y, test_labels, background_cost)
print(f"Question 3e (cost = {background_cost}):")
print(f"True Positive Rate = {tpr_costly:.4f}")
print(f"False Positive Rate = {fpr_costly:.4f}")

# ROC Curve and AUC calculation
conf_scores = log_p_y[:, 1] - log_p_y[:, 0]
area_under_roc, alarm_rate, detect_rate = area_roc(conf_scores, test_labels, do_plot=True)
print(f"\nArea Under ROC Curve = {area_under_roc:.4f}")