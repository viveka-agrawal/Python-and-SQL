
# UCI CS275P: Utilities for training binary VAE models

import numpy as np
import matplotlib.pyplot as plt

# Pytorch Includes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



class BinaryMNIST(torch.utils.data.Dataset):
	''' 
		This is the basic MNIST dataset but instead of continuous pixel values
		we convert the images to binary images where each pixel is either 0 or 1.
	'''
	def __init__(self):
		# Get the MNSIT dataset, this is the normal MNIST.  We will binarize it.
		self.mnist_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)

	def __len__(self):
		return len(self.mnist_dataset)

	def __getitem__(self, idx):
		img, label = self.mnist_dataset[idx]

		# Remove the unneeded first dim
		img = img.squeeze(0)

		# convert to a binary image
		img = self._binarize_img(img)

		return img, label

	def _binarize_img(self, imgs):
		gt_half = imgs > 0.5

		imgs[gt_half] = 1.0
		imgs[~gt_half] = 0.0

		return imgs



class Encoder(nn.Module):
	def __init__(self, latent_dim, input_dim=784, hidden_dim=250):
		''' 
			The encoder part of the VAE model.

			Parameters:
				latent_dim: The size of the latent space (number of dimensions)
				input_dim: The size of the input of the Encoder (aka image size)
				hidden_dim: The size of the hidden dimensions for the fully connected layers
		''' 
		super(Encoder, self).__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc_mu = nn.Linear(hidden_dim, latent_dim)
		self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)

		mu = self.fc_mu(x)
		sigma = self.fc_sigma(x)
		sigma = F.relu(sigma) + 1e-4

		return mu, sigma

class Decoder(nn.Module):
	def __init__(self, latent_dim, output_dim=784, hidden_dim=250):
		''' 
			The decoder part of the VAE model

			Parameters:
				latent_dim: The size of the latent space (number of dimensions)
				output_dim: The size of the output of the Decoder (aka image size)
				hidden_dim: The size of the hidden dimensions for the fully connected layers
		''' 
		super(Decoder, self).__init__()
		self.fc1 = nn.Linear(latent_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x

class VAE(nn.Module):
	def __init__(self, latent_dim, in_out_dim=784, hidden_dim=250):
		''' 
			The VAE model

			Parameters:
				latent_dim: The size of the latent space (number of dimensions)
				in_out_dim: The size of the input and output of the VAE (aka image size)
				hidden_dim: The size of the hidden dimensions for the fully connected layers
		''' 

		super(VAE, self).__init__()

		self.latent_dim = latent_dim

		self.encoder = Encoder(latent_dim, in_out_dim, hidden_dim=hidden_dim)
		self.decoder = Decoder(latent_dim, in_out_dim, hidden_dim=hidden_dim)

	def forward(self, x):

		# Encode
		mu, sigma = self.encoder(x)

		# Sample
		z = self.draw_sample_from_dist(mu, sigma)

		# Decode
		output = self.decoder(z)

		# Make into valid probability distribution
		output = torch.sigmoid(output)

		# Return everything
		return output, mu, sigma


	def draw_sample_from_dist(self, mu, sigma):

		# Draw samples from the N(0,1) distribution
		norm_rands = torch.randn_like(mu)

		# Convert samples from Normal Distribution to the one predicted 
		# by the encoder.  Note: This is mean-field
		z = (norm_rands*sigma) + mu

		return z


def elbo_loss_function(reconstructed_img, true_img, z_mu, z_sigma):
	''' Compute the ELBO based loss function that we want to minimize.
		Note: This is basically the -ELBO

		Parameters:
			reconstructed_img: Bernoulli parameters of the reconstruction image
			true_img: The ground truth image
			z_mu: Latent space mean
			z_sigma: Latent space std

		Returns:
			elbo: The ELBO based loss
	'''

	# https://arxiv.org/pdf/1312.6114.pdf

	# Reconstruction Error
	recon_error = F.binary_cross_entropy(reconstructed_img, true_img, reduction='none')
	recon_error = torch.mean(torch.sum(recon_error, -1))

	# KL-Divergence of latent space distribution to N(0,1)
	kl_div = 0.5 * torch.sum(1 + 2*torch.log(z_sigma) - (z_mu**2)  - (z_sigma**2), -1)
	kl_div = torch.mean(kl_div)

	# Overall loss is just the sum of these
	elbo = recon_error - kl_div
	return elbo



def plot_2d_clusterings(model, dataset, title):
	''' 
		Plot "Clustering" (note here we dont actually do any clustering algorithms but instead just plot each point with a color)
		using the latent codes

		Parameters:
			model: The model to use to get the latent codes
			dataset: The dataset to cluster.
			title: Latent space mean

		Returns:
			fig: The figure 
	'''

	# Use the loader to get 1 BIG batch. aka all the data in 1 batch
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True, num_workers=6)

	# Extract the images from the data tuple, we DO care about the labels 
	imgs, labels = next(iter(loader))

	# Flatten the images into a flat vector
	imgs = torch.reshape(imgs, (imgs.shape[0], -1))

	# convert to numpy for fast plotting
	labels = labels.numpy()

	if(isinstance(model , nn.Module)):

		# Move to the CPU just in case
		model_internal = model.cpu()

		with torch.no_grad():
			model_internal.eval()
			
			# Get the means!
			latent, _ = model_internal.encoder(imgs)
			latent = latent.numpy()

	else:
		# ppca only works on cpu
		imgs = imgs.numpy()

		# Encode
		latent = model.encode(imgs)


	# Create a new figure
	fig = plt.figure()

	# Plot the different classes
	cmap = plt.cm.get_cmap()
	classes = np.unique(labels)
	cvals = (classes - np.min(classes)) / (np.max(classes)-np.min(classes)+1e-100)
	for i,c in enumerate(classes): 
		# plt.scatter(latent[labels==c,0],latent[labels==c,1], edgecolors="black", color=cmap(cvals[i]), s=1)  
		plt.scatter(latent[labels==c,0],latent[labels==c,1], color=cmap(cvals[i]), s=1)  


	fig.suptitle(title)

	return fig 
