
# UCI CS275P: EM Algorithm for Factor Analysis from sparse observations

import numpy as np

def sparse_fa_em (s_train, K, max_iters, verbose=False):
	''' Do EM Algorithm for Sparse Factor Analysis

		Parameters:
	     s_train   - #movies x #users sparse matrix of movie ratings
	     K         - integer number of latent factors (size of low-dim space)
	     max_iters - number of iterations to run EM
	     verbose   - If we should be verbose (default False)
	  
	  	Returns:
			r_train   - low-dim representation of each user (#users X K)
			w         - factor loading matrix (#movies X K)
			xbar      - mean ratings for each movie (#movies X 1)
			psi       - vector of rating variances for each movie (#movies x 1)
			lls       - [list] marginal log-likelihood bound at each iteration

	'''
	conv_thresh = 0
	eps = 1e-8

	s_train = s_train.astype("float")

	# Extract some info
	num_movies =  s_train.shape[0]
	num_users = s_train.shape[1]

	average_r = np.sum(s_train, 1)
	for i in range(num_movies):
		average_r[i] = average_r[i].astype(float) / np.sum(s_train[i]>0).astype("float")

	# initialize
	w = np.sqrt(10) * np.random.randn(num_movies, K) # init transform
	psi = np.ones((num_movies, )) 					 # diagonal covariance matrix
	xbar = np.copy(average_r)						 # mean of observed ratings


	######################################################################
	# EXPECTATION MAXIMIZATION ALGORITHM
	######################################################################
	done = False
	iters = 0
	lls = []
	loglike = -1.0 / eps

	while(done == False):

		iters += 1
		loglike_old = loglike

		# init iteration
		psi_new = np.zeros((num_movies,))
		ez = np.zeros(( K+1, num_users ))
		ezz = np.zeros(( K+1, K+1, num_users ))
		Q = np.zeros(( K, K, num_users ))
		w_top = np.zeros(( K+1, num_movies ))  
		w_bottom = np.zeros(( K+1, K+1, num_movies ))



		#############################################################
		# E-Step
		#############################################################
		for n in range(num_users):
			x_n = s_train[:, n]
			I = x_n != 0
			if(np.sum(I) == 0):
				continue

			# construct observed quantities
			w0 = w[I,:]
			inv_psi0 = np.diag(1. / psi[I])
			x0 = x_n[I]
			xbar0 = xbar[I]

			# compute sufficient statistics     
			Q[:,:,n] = np.linalg.inv( np.eye(K) + (w0.T @ inv_psi0 @ w0))
			m = Q[:,:,n] @ (w0.T @ inv_psi0 @ np.reshape((x0 - xbar0), (-1, 1)))
			V = Q[:,:,n] + (m @ m.T)
			ez[:,n] = np.append(m, 1)
			ezz[:-1,:-1,n] = V
			ezz[:-1,-1,n] = m[:, 0]
			ezz[-1,:-1,n] = m[:, 0]
			ezz[-1,-1,n] = 1.0


			# compute terms used to update W
			w_top[:,I] = w_top[:,I] + np.reshape(ez[:,n], (-1, 1)) @ np.reshape(x0.T, (1, -1))
			w_bottom[:,:,I] = w_bottom[:,:,I] + np.atleast_3d(ezz[:,:,n])


		#############################################################
		# M-Step
		#############################################################
		w_aug = np.zeros((num_movies, K+1))
		for m in range(num_movies):
			w_aug[m,:] = np.linalg.lstsq(w_bottom[:,:,m], w_top[:, m], rcond=None)[0]

		w_new = w_aug[:, :-1]
		xbar_new = w_aug[:, -1]
		t_tot = np.zeros((num_movies,))


		for n in range(num_users):
			x_n = s_train[:, n]
			tn = (x_n != 0).astype("float")
			t_tot = t_tot + tn

			pred_error = (x_n - (w_aug @ ez[:, n]))**2 + np.sum( (w_new @ Q[:, :, n]) * w_new, -1)
			psi_new = psi_new + (tn * pred_error)

		psi_new = psi_new / t_tot

		# update parameters
		w = w_new
		psi = psi_new
		xbar = xbar_new	


		#############################################################
		# Compute complete data log-likelihood and Stop Conditions
		#############################################################
		ll_this = 0
		H = 0		
		for n in range(num_users):
			x_n = s_train[:,n]
			tn = (x_n != 0).astype("float")

			ll_this = ll_this + np.sum(np.diag(ezz[:,:,n]))
			H = H + (K/2.0) * np.log( 2.0*np.pi*np.e) + (0.5 * np.log( np.linalg.det( Q[:,:,n])))

			pred_error = (x_n - w_aug@ez[:,n])**2 + np.sum((w_new@Q[:,:,n]) * w_new, -1)
			ll_this = ll_this + np.reshape(tn, (1, -1))  @ (np.log(psi) + (pred_error/psi))


		loglike = -0.5 * ll_this + H
		lls.append(loglike)
		diff_loglike = loglike - loglike_old;
		
		if(verbose):
			print(iters, diff_loglike)

		# check for convergence
		if(iters == max_iters):
			done = True
		elif ((iters > 1 ) and (diff_loglike <= conv_thresh)):
			done = True


	#############################################################
	# EVALUATE TEST DATA
	#############################################################

	r_train = np.zeros((num_users,K))

	for n in range(num_users):
		x_n = s_train[:,n]
		I = x_n != 0
		if(np.sum(I) == 0):
			continue

		# construct observed quantities
		w0 = w[I,:]
		inv_psi0 = np.diag(1. / psi[I])
		x0 = x_n[I]
		xbar0 = xbar[I]

		# compute posterior mean
		G = np.linalg.inv( np.eye(K) + w0.T @ inv_psi0 @ w0 );
		ez = G @ (w0.T @ inv_psi0 @ np.reshape((x0 - xbar0), (-1, 1)))

		# predict rankings
		r_train[n,:] = np.reshape(ez, (-1,))

	return r_train, w, xbar, psi, lls
