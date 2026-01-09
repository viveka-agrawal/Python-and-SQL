import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import torch

# pip3 install pomegranate
# or 
# conda install conda-forge::pomegranate

from pomegranate.distributions import Normal

# Uses sklearn to find the starting cluster points
def create_initial_clusters(x, k, method="k-means++"):
    '''
        Create an initial clustering using an initialize method from sklean KMeans

        Parameters:
            x: data to cluster
            k: number of clusters
            method: "random" or "k-means++", the method for the initial clustering
        
        Returns:
            Cluster assignments for each x
    '''
    kmeans = KMeans(n_clusters=k, init=method, max_iter=1, n_init=1)
    kmeans.fit(x)
    return kmeans.predict(x)

# Creates the initial cluster distributions for pomegranate
def create_initial_cluster_distributions(x, k, cluster_assignments):
    '''
        Create an the initial cluster distributions from cluster assignments

        Parameters:
            x: data to cluster
            k: number of clusters
            cluster_assignments: The cluster assignments for x
            # is_diagonal: if the multivariate Gaussian distribution has a diagonal covariance matrix. If false then it is unconstrained.
            
        Returns:
            Distribution object
    '''

    # Create the initial Gaussian components
    components = []
    for c in range(k):
        
        # Extract only the samples in the clusters
        cluster_k_xs = x[cluster_assignments==c]
        
        # Create the distribution from the samples.  This computes the mean and 
        # covariance matrix from the samples
        if(False):
            # Create a distribution per dimention and then combine them.  This is the same
            # as setting a diagonal matrix but this is how pomegranate needs it to be
            
            dists = [NormalDistribution.from_samples(np.expand_dims(cluster_k_xs[:, d], -1))  for d in range(x.shape[-1])]
            dist = IndependentComponentsDistribution(dists) 
        else:
            # dist = MultivariateGaussianDistribution.from_samples(cluster_k_xs)
            dist = Normal()
            dist.fit(cluster_k_xs)
        components.append(dist)
    return components

def plot_clustered_images(model, pca_features, images, num_clusters_to_plot=10,num_images_per_cluster=7):
    '''
        Show the top images for some clusters (aka the ones with the highest probability per cluster)

        Parameters:
            model: The GMM to sample images from (aka the model with the clusters in it)
            pca_features: The PCA features (so we can compute log_prob per cluster)
            images: The raw images (so we can plot them)
            num_clusters_to_plot: Max Number of clusters to plot, could plot less if model has less clusters
            num_images_per_cluster: Number of images to plot per cluster

        Return:
            matplotlib figure object

    '''

    # Assign each image to a cluster
    predictions = model.predict(pca_features)

    # Compute the log prob for each cluster for each image
    log_probs = model.predict_log_proba(pca_features)
        
    # compute how many we should plot, can only plot if the model has that 
    # many clusters
    num_clusters_in_model = log_probs.shape[-1]
    num_clusters_to_plot = min(num_clusters_in_model, num_clusters_to_plot)

    # PLOT PLOT PLOT!!!!
    rows = num_images_per_cluster
    cols = num_clusters_to_plot
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False,  figsize=(8,8))

    for c in range(num_clusters_to_plot):

        # Extract the images and their log probabilities for this cluster
        cluster_log_probs = log_probs[predictions == c][:, c]
        cluster_imgs = images[predictions == c]

        # Find the images with the top probabilities for this cluster 
        indices = np.argpartition(cluster_log_probs, num_images_per_cluster)[-num_images_per_cluster:]

        for i in range(num_images_per_cluster):

            # Extract the image to plot
            img = cluster_imgs[indices[i]]

            # Plot!!
            ax = axes[i, c]
            ax.imshow(img)
            ax.axis("off")

            # Format the plot to look nice
            if(i == 0):
                ax.set_title("cluster {:d}".format(c))

    fig.suptitle("Clustered Images (k={:d})".format(num_clusters_in_model),fontweight ="bold")
    fig.tight_layout()

    return fig

def plot_sampled_images(model, pca, num_clusters_to_plot=10, num_images_per_cluster=7):
    '''
        Sample images from each cluster and plot them.

        Parameters:
            model: The GMM to sample images from (aka the model with the clusters in it)
            pca: The sklearn PCA object used to compute the PCA features.  Used to transform from latent space to image space
            num_clusters_to_plot: Max Number of clusters to plot, could plot less if model has less clusters
            num_images_per_cluster: Number of images to plot per cluster

        Return:
            matplotlib figure object
    '''


    # compute how many we should plot, can only plot if the model has that 
    # many clusters
    num_clusters_in_model = len(model.distributions)
    num_clusters_to_plot = min(num_clusters_in_model, num_clusters_to_plot)

    # PLOT PLOT PLOT!!!!
    rows = num_images_per_cluster
    cols = num_clusters_to_plot
    # fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(12,12))
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False,  figsize=(8,8))

    for c in range(num_clusters_to_plot):
        # Get all the distribution for this cluster
        cluster_model = model.distributions[c]

        mean = cluster_model.means
        scale_tril = torch.linalg.cholesky(cluster_model.covs)
        cluster_dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril=scale_tril)

        # def is_psd(mat):
        #     # return bool((mat == mat.T).all()) #and (torch.eig(mat)[0][:,0]>=0).all())
        #     # return bool((torch.linalg.eig(mat)[0][:,0]>=0).all())

        #     diff = mat - mat.T
        #     print(diff)
        #     exit()

        # print(cluster_model.covs.shape)
        # print(is_psd(cluster_model.covs))
        # exit()

        for i in range(num_images_per_cluster):

            # Generate a sample from the latent space
            sample = np.atleast_1d(cluster_dist.sample((1,)))

            # Convert that latent space sample into a complete image
            sampled_image = pca.inverse_transform(sample)
            sampled_image = np.reshape(sampled_image, (32,32,3))

            # Clip to [0, 255] to make it a valid image
            # Also convert to uint8 so we can render it
            sampled_image[sampled_image < 0] = 0
            sampled_image[sampled_image > 255] = 255
            sampled_image = sampled_image.astype("uint8")

            # Plot the Image
            ax = axes[i, c]
            ax.imshow(sampled_image)
            ax.axis("off")

            # Only put a cluster title on the first row
            if(i == 0):
                ax.set_title("cluster {:d}".format(c))

    fig.suptitle("Samples from Clusters (k={:d})".format(num_clusters_in_model),fontweight ="bold")
    fig.tight_layout()

    return fig



