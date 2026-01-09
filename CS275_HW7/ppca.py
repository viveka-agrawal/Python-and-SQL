
# UCI CS275P: Probablistic PCA training from dense observations

import numpy as np
import scipy.sparse.linalg
from scipy import special
from numpy.random import randn
from numpy.matlib import repmat


class PPCA:
    def __init__(self, latent_dim = 2, verbose = False):
        '''
            Constructor

            Parameters:
                latent_dim: The size of the latent dimension (latent codes)
                verbose: If this object should be verbose in its output
        '''

        self.latent_dim = latent_dim
        self.verbose = verbose
        

        self.mean = None
        self.V = None
        self.var = None
        
    def fit(self, data):
        '''
            Estimate the parameters of this PPCA object from data. 

            Parameters:
                data: The data to fit this PPCA with.  Shape: [N, D] where N is the number of data samples and D is 
                      the dimension of the data
        '''

        mean = np.mean(data.T, axis=1)

        # compute mean (is a vector of means per variable)
        mean = np.mean(data.T, axis=1)
        if(self.verbose):
            print('[Training] computed mean ' + 'x'.join(map(str, mean.shape)))

        # center
        means = np.repeat(mean.reshape((1, mean.shape[0])), data.shape[0], axis = 0)
        data = data - means
        if(self.verbose):
            print('[Training] centered data')

        # We need all the eigenvectors and values ...
        U, s, Vt = scipy.sparse.linalg.svds(data, k=self.latent_dim)
        if(self.verbose):
            print('[Training] computed first ' + str(self.latent_dim) + ' singular vectors')

        approximate_k = 300
        # approximate_k = 10
        # approximate_k = data.shape[1] - self.latent_dim
        approximate_k = min(approximate_k, data.shape[0])
        _, s_all, _ = scipy.sparse.linalg.svds(data, k=approximate_k)
        if(self.verbose):
            print('[Training] computed first ' + str(approximate_k) + ' singular values')

        # singular values to eigenvalues
        e = s**2/(data.shape[0] - 1)
        e_all = s_all**2/(data.shape[0] - 1)

        # compute variance
        var = 1.0/(data.shape[0] - self.latent_dim)*(np.sum(e_all) - np.sum(e))
        if(self.verbose):
            print('[Training] variance ' + str(var) + ' (' + str(np.sum(e_all))  + ' / ' + str(np.sum(e)) + ')')

        # compute V
        L_m = np.diag(e - np.ones((self.latent_dim))*var)**0.5
        V = Vt.T.dot(L_m)


        self.mean = mean
        self.V = V
        self.var = var

    def recover(self, data):
        '''
            Reconstruct the input (data) using PPCA.  This is done by encoding the data into the lower 
            dimensional latent codes and then decoding it back into the original data dimension. This 
            "recovers" the original data after compressing and decompressing it.

            This is equivalent to doing the following:
                reconstructed_x = self.decode(self.encode(x))

            Parameters:
                data: The data to reconstruct.  Shape: [B, D] where B is the batch and D is 
                      the dimension of the data

            Returns:
                The reconstructed data. Shape [B, D]
        '''

        if(self.mean is None):
            print("Must call fit first on PPCA")
            assert(False)

        mean = self.mean
        V = self.V
        var = self.var

        I = np.eye(V.shape[1])
        M = V.T.dot(V) + I*var
        M_inv = np.linalg.inv(M)

        means = np.repeat(mean.reshape((mean.shape[0], 1)), data.shape[0], axis = 1)
        codes = M_inv.dot(V.T.dot(data.T - means))

        code_mean = np.mean(codes)
        code_var = np.var(codes)
        if(self.verbose):
            print('[Validation] codes: ' + str(code_mean) + ' / ' + str(code_var))


        # print(means)
        # exit()

        preds = np.dot(V, codes) + means
        preds = preds.T

        return preds



    def encode(self, data):
        '''
            Encode the data into the latent codes using the fit PPCA. (Compression)

            Parameters:
                data: The data to encode.  Shape: [B, D] where B is the batch size and D is 
                      the dimension of the data

            Returns:
                The encoded latent codes. Shape [B, latent_dim]
        '''

        if(self.mean is None):
            print("Must call fit first on PPCA")
            assert(False)


        mean = self.mean
        V = self.V
        var = self.var

        I = np.eye(V.shape[1])
        M = V.T.dot(V) + I*var
        M_inv = np.linalg.inv(M)

        means = np.repeat(mean.reshape((mean.shape[0], 1)), data.shape[0], axis = 1)
        codes = M_inv.dot(V.T.dot(data.T - means))

        return codes.T



    def decode(self, codes):
        '''
            Decode latent codes.  This takes the latent codes and decodes them back into the origination 
            data dimensions using the fit PPCA. (Decompression)

            Parameters:
                codes: The encoded latent codes. Shape [B, latent_dim] where B is the batch size

            Returns:
                The decoded predictions in the dim of the original data. Shape: [B, D] 
                    where B is the batch size and D is the dimension of the data
        '''


        if(self.mean is None):
            print("Must call fit first on PPCA")
            assert(False)

        mean = self.mean
        V = self.V

        means = np.repeat(mean.reshape((mean.shape[0], 1)), codes.shape[0], axis = 1)
        preds = np.dot(V, codes.T) + means
        preds = preds.T

        return preds
