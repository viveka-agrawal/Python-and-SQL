
# UCI CS275P: VAE models for binary MNIST digits
# This is DEMONSTRATION code:
# - It gives helpful examples of how to load the binary MNIST digit data,
#   and train PPCA and VAE models.
# - It shows how to plot reconstructions of images given learned low-dim. embeddings.
# - It shows how to sample synthetic images from trained PPCA/VAE models.
# - You may (but do not have to) reuse parts of this code in your solutions. 
# - It is NOT a template for the individual questions you must answer,
#   for that see the main homework pdf.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm

# Pytorch Includes
import torch
import torch.optim as optim

# Homework Includes
from vae_utils import *
from ppca import *

# If you have a GPU then you can train on it by setting this to true
train_on_gpu = False 

# Dataset we will be using for all parts 
train_dataset = BinaryMNIST()

##############################################################################################################
## Probabilistic PCA with 2 dim latent space
##############################################################################################################

# Get all the data for the PPCA
loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=6)
all_imgs, all_labels = next(iter(loader))

# Flatten the images into a flat vector
all_imgs = torch.reshape(all_imgs, (all_imgs.shape[0], -1))
all_imgs = all_imgs.numpy()

# Create the PPCA and fit it
print("Fitting PPCA Model with 2 Dim Latent Space")
ppca = PPCA(latent_dim=2, verbose=False)
ppca.fit(all_imgs)

####################################################
# Draw a sample
####################################################

# Sample some latent space values
z = np.random.randn(1, ppca.latent_dim)

# Generate images from them
img = ppca.decode(z)
img[img < 0] = 0
img[img > 1] = 1

# Reshape the images to be images 
img = np.reshape(img, (28, 28))

fig = plt.figure()
plt.imshow(img, cmap="Greys", vmin=0, vmax=1)
plt.title("PPCA Sample")

####################################################
# Reconstruct Image
####################################################
orig_img, _ = train_dataset[0]
flat_orig_img = orig_img.flatten()

# Convert to numpy and make sure its 2D
flat_orig_img = flat_orig_img.numpy()
flat_orig_img = np.reshape(flat_orig_img, (1, -1))


# Create a reconstruction
recon_img = ppca.recover(flat_orig_img)
recon_img = np.reshape(recon_img, ( 28, 28))

fig = plt.figure()
plt.imshow(recon_img, cmap="Greys", vmin=0, vmax=1)
plt.title("PPCA Reconstruction")

####################################################
# Cluster
####################################################
fig_2d_clustering_ppca = plot_2d_clusterings(ppca, train_dataset, "PPCA 2 Dim latent Space Clustering")


# ##############################################################################################################
# ## VAE with 2 dim latent space
# ##############################################################################################################

# Create the VAE 2 dim latent space
vae = VAE(latent_dim=2)

# Train the model
print("Training VAE Model with 2 Dim Latent Space")

# If we are using the GPU then use this
if(train_on_gpu):
	device = "cuda"
else:
	device = "cpu"

# Create the training loader needed for pytorch.  This loader will load the data from the dataset and return 
# batches of data to us. We also automatically shuffle the data
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

# Move the vae to the correct device
vae = vae.to(device)

# Create the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Keep track of the losses
training_losses = []

# Train!
num_epochs = 25
for epoch in tqdm(range(num_epochs)):

	all_epoch_losses = []
	
	# Go through all the data once
	t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
	for step, data in enumerate(t):

		# Extract the images from the data tuple, we dont care about the labels 
		imgs, _ = data

		# Flatten the images into a flat vector
		imgs = torch.reshape(imgs, (imgs.shape[0], -1))

		# Move data to the correct device
		imgs = imgs.to(device)

		# Zero out the gradients for the model
		vae.zero_grad()

		# Pass data through the VAE
		recon_imgs, mu, sigma = vae(imgs)

		# Compute the loss
		loss = elbo_loss_function(recon_imgs, imgs, mu, sigma)
		all_epoch_losses.append(loss.detach().item())

		# Compute the gradients
		loss.backward()

		# Take a gradient step
		optimizer.step()

	avg_epoch_loss = sum(all_epoch_losses) / len(all_epoch_losses)
	training_losses.append(avg_epoch_loss)

training_losses = np.asarray(training_losses)

# Plot the loses
fig_2d_losses = plt.figure()
plt.plot(training_losses)
plt.xlabel("Epoch")
plt.ylabel("ELBO Loss")
plt.title("VAE 2 Dim latent Space Training Losses")


####################################################
# Draw a sample
####################################################
with torch.no_grad():
	vae.eval()

	# Move to the CPU just in case
	vae = vae.cpu()

	# Sample some latent space values
	z = torch.randn((1, vae.latent_dim))

	# Generate images from them
	img = vae.decoder(z)
	img = torch.sigmoid(img)

	# Reshape the images to be images 
	img = torch.reshape(img, (28, 28))

	# Move to numpy 
	img = img.numpy()

	fig = plt.figure()
	plt.imshow(img, cmap="Greys", vmin=0, vmax=1)
	plt.title("VAE Sample")


####################################################
# Reconstruct Image
####################################################
with torch.no_grad():
	vae.eval()

	# Move to the CPU just in case
	vae = vae.cpu()

	orig_img, _ = train_dataset[0]
	flat_orig_img = orig_img.flatten()

	# Make sure its 2d
	flat_orig_img = flat_orig_img.unsqueeze(0)	

	# Create a reconstruction
	recon_img, _, _ = vae(flat_orig_img)
	recon_img = torch.reshape(recon_img, (28, 28))

	fig = plt.figure()
	plt.imshow(recon_img.numpy(), cmap="Greys", vmin=0, vmax=1)
	plt.title("VAE Reconstruction")


####################################################
# Cluster
####################################################
fig_2d_clustering = plot_2d_clusterings(vae, train_dataset, "VAE 2 Dim latent Space Clustering")


plt.show()

