
# UCI CS275P: Utilities for movie rating data

import numpy as np
from sklearn.decomposition import PCA

def arrays_same_shape(a, b):
	'''
		Makes sure that 2 arrays have the same shape

		Parameters:
			a: array a
			b: array b

		Return:
			true: if a and b have the same shape, False otherwise
	'''

	if(len(a.shape) != len(b.shape)):
		return False

	for i in range(len(a.shape)):
		if(a.shape[i] != b.shape[i]):
			return False

	return True

# def pca(A):
# 	""" 
# 	Taken From: https://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html

# 	Performs principal components analysis 
# 	 (PCA) on the n-by-p data matrix A
# 	 Rows of A correspond to observations, columns to variables. 

# 	Returns :  
# 		coeff : is a p-by-p matrix, each column containing coefficients for one principal component.
# 		score : the principal component scores; that is, the representation of A in the principal 
# 				component space. Rows of SCORE correspond to observations, columns to components.
# 		latent : a vector containing the eigenvalues of the covariance matrix of A.
# 	"""
# 	# computing eigenvalues and eigenvectors of covariance matrix
# 	M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
# 	[latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
# 	score = np.dot(coeff.T,M) # projection of the data in the new space
# 	return coeff, score, latent

def pca(A):
	""" 
	Performs principal components analysis (PCA) on the n-by-p data matrix A.
	Rows of A correspond to observations, columns to variables. 

	Returns :  
		score (n_features, n_components): the principal component scores; 
          that is, the representation of A in the principal component space. 
          Rows of SCORE correspond to observations, columns to components.
		coeff (n_samples, n_components): is a p-by-p matrix, 
          each column containing coefficients for one principal component.
		mu (n_features,): np.ndarray of means for each observation dimension 
	"""
	pca = PCA()
	pca.fit(A)
	coeff = np.transpose(pca.components_)
	score = pca.fit_transform(A)
	mu = np.mean(A, axis=0)
	return score, coeff, mu

def calc_rmse_for_movie_ratings(s_true, s_hat):

	# Need to have the same shape!!
	if(arrays_same_shape(s_true, s_hat) == False):
		print("s_true and s_hat must be the same shape")
		assert(False)

	sum_squared_dist = 0.0
	for n in range(s_true.shape[1]):
		non_zero_idxs = s_true[:, n] != 0
		squared_dist = np.sum((s_true[non_zero_idxs, n] - s_hat[non_zero_idxs, n])**2)
		sum_squared_dist += squared_dist

	n_test = np.sum(s_true>0).astype("float")
	rmse = np.sqrt(sum_squared_dist / n_test)

	return rmse
