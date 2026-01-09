
# Standard imports 
import numpy as np

def basis_poly(x, order):
    ''' Generates a polynomial basis expansion of x for a specific order

        Parameters:
            x: (N,) array of values to compute the basis functions for
            order: [Integer] Order of the basis function

        Returns
            (N, order+1) array of the basis function for each input in x
    '''
    phi = np.zeros(shape=(x.shape[0], order+1))
    for p in range(order+1):
        phi[:, p] = x**p

    return phi


def basis_radial(x, order):
    ''' Generates a radial basis expansion of x for a specific order

        Parameters:
            x: (N,) array of values to compute the basis functions for
            order: [Integer] Order of the basis function

        Returns
            (N, order+1) array of the basis function for each input in x
    '''

    # Some info about the problem
    N = x.shape[0]
    M = order+1

    phi = np.zeros((N,M))
    phi[:,0] = 1.0
    
    if(order > 1):
        centers   = np.linspace(-1, 1, order)
        bandwidth = centers[1] - centers[0]
    elif(order == 1):
        centers   = 0
        bandwidth = 1
    else:
        return phi

    for i in range(order):
        phi[:,i+1] = np.exp(-0.5 * ((x - centers[i]) / bandwidth)**2);

    return phi


def generate_basis_matrices(x_train_rescaled, x_test_rescaled, x_grid_rescaled, basis_func, order):
    """
    Generate basis function matrices for training, test, grid, and held-out data.
    
    Args:
        x_train_rescaled: Rescaled training input data
        x_test_rescaled: Rescaled test input data
        x_grid_rescaled: Rescaled grid of input points (to be used for plots)
        basis_func: Basis function to use (basis_poly or basis_radial)
        order: Order of the basis function
        
    Returns:
        phi_train, phi_test, phi_grid: Basis matrices
        M: Number of basis functions
    """
    M = order + 1
    
    # Compute the basis matrices
    phi_train = basis_func(x_train_rescaled, order)
    phi_test = basis_func(x_test_rescaled, order)
    phi_grid = basis_func(x_grid_rescaled, order)
    
    return phi_train, phi_test, phi_grid, M


def rescale_inputs(x_train, x_test, x_grid):
    """
    Rescale inputs for numerical stability.
    
    Parameters:
    ----------
    x_train : numpy.ndarray
        Training input features
    x_test : numpy.ndarray
        Testing input features
    x_grid : numpy.ndarray
        Regular grid of inputs (to be used for plots)
        
    Returns:
    -------
    x_train_rescaled : numpy.ndarray
        Rescaled training input features
    x_test_rescaled : numpy.ndarray
        Rescaled testing input features
    x_grid_rescaled : numpy.ndarray
        Rescaled grid of input features
    x_offset : float
        Offset used for rescaling
    x_scale : float
        Scale used for rescaling
    """
    x_offset = (np.min(x_train) + np.max(x_train)) / 2.0
    x_scale = (np.max(x_train) - np.min(x_train)) / 2.0
    x_train_rescaled = (x_train - x_offset) / x_scale
    x_test_rescaled = (x_test - x_offset) / x_scale
    x_grid_rescaled = (x_grid - x_offset) / x_scale
    
    return x_train_rescaled, x_test_rescaled, x_grid_rescaled, x_offset, x_scale

