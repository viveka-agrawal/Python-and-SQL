import time
import numpy as np
import torch

from pomegranate.distributions import Bernoulli

class FixedBernoulli(Bernoulli):
	"""A Bernoulli distribution object.

	A Bernoulli distribution models the probability of a binary variable
	occurring. rates of discrete events, and has a probability parameter
	describing this value. This distribution assumes that each feature is 
	independent of the others.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probablity parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	probs: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The probability parameters for each feature. Default is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	"""

	def __init__(self, probs=None, inertia=0.0, frozen=False, check_data=True):
		super().__init__(probs=probs, inertia=inertia, frozen=frozen, check_data=check_data)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self.register_buffer("_w_sum", torch.zeros(self.d, device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(self.d, device=self.device))

		# Compte the probablity and its inverse
		p = self.probs
		p_inv = 1 - p
		
		# Make numerically stable
		p_eps = 1e-12
		p = p + p_eps
		p_inv = p_inv + p_eps		

		self.register_buffer("_log_probs", torch.log(p))
		self.register_buffer("_log_inv_probs", torch.log(p_inv))


	def get_p_values(self):
		return self.probs

	def get_p_values_numpy(self):
		return self.probs.detach().cpu().numpy()


def create_components(x, k):
    """
    Create initial components for a Bernoulli Mixture Model.

    Args:
        x (numpy.ndarray): Input data matrix where rows are samples and columns are features.
        k (int): Number of clusters to create.

    Returns:
        list: A list of FixedBernoulli distributions initialized with random parameters.
    """
    # Randomize the order of the images and the labels since they are not shuffled
    rand_indices = list(range(x.shape[0]))
    np.random.shuffle(rand_indices)
    x = x[rand_indices]

    # Create the components initially via random initialization using the data
    components = []
    split_size = x.shape[0] // k
    for k in range(k):

        # Extract some data for the split
        s = k*split_size
        e = s+split_size
        data = x[s:e]

        # Compute the MLE for the Bernoulli from data.  
        # Add random pseudo-count to the value and clamp it to be in range
        p = np.mean(data, axis=0)
        p += np.random.randn(p.shape[0]) * 0.1
        p[p < 0] = 0.0
        p += 0.001
        p[p > 1] = 1.0

        # Create the distribution
        dist = FixedBernoulli(probs=p)
        components.append(dist)

    return components


