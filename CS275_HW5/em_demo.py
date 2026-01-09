# UCI CS275P: Gaussian Mixture Models Expectation-Maximization Demo
# This is DEMONSTRATION code:
# - It gives helpful examples of how to load the sun397 data and run EM algorithm
# - It shows how to use the plotting functions provided to by em_utils.py
# - You may (but do not have to) reuse parts of this code in your solutions. 
# - It is NOT a template for the individual questions you must answer,
#   for that see the main homework pdf.

# standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch

# Requires pomegranate package implementing EM for mixture models:
# pip3 install pomegranate
# or 
# conda install conda-forge::pomegranate

# Homework imports
from general_utils import *
from em_utils import *

# If we should use the GPU to fit the model
use_gpu = False

# Load the data
data = np.load("subset_of_sun397_tiny_dataset_with_pca.npy", allow_pickle=True).item()
images = data["images"]
labels = data["labels"]
pca_features = data["pca_features"]

# Convert to pytorch floats
pca_features = torch.from_numpy(pca_features).float()
labels = torch.from_numpy(labels).float()

# Load the "trained" PCA 
pca = pk.load(open("pca.pkl",'rb'))

# Try with 3 clusters
k = 3

# Create the initial cluster distributions
initial_cluster_assignments = create_initial_clusters(pca_features, k)
components = create_initial_cluster_distributions(pca_features, k, initial_cluster_assignments)

# Create the GMM object we will fit
model = CustomGeneralMixtureModel(components, verbose=True, tol=1e-8)

# Do the EM algorithm
if(use_gpu):

    # Move the model to the GPU
    model = model.cuda()

    # Move the features to the GPU and then fit
    _, history = model.fit(pca_features.cuda())

    # Move the model back to the CPU so the rendering code can work
    model = model.cpu()
else:

    # Just fit without any GPU/CPU transfers ext
    model.fit(pca_features)        

# # Plot!
fig0 = plot_clustered_images(model, pca_features, images)
fig1 = plot_sampled_images(model, pca)

plt.show()
