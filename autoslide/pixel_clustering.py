"""
Cluster pixels in an image
"""
import os
from glob import glob
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm
from tqdm import tqdm, trange
from pprint import pprint as pp
import autoslide
from autoslide.pipeline.utils import get_threshold_mask

from autoslide import config

# Get data directory from config
data_dir = config['data_dir']

img_dir = os.path.join(data_dir, 'images') 
img_list = sorted(glob(os.path.join(img_dir, '*')))

ind = 8
img_path = os.path.join(data_dir, 'images', 'sorted_8.png')
img = plt.imread(img_path) 
# img = plt.imread(img_list[ind]) 
img = img[...,:3]

plt.imshow(img)
plt.show()

img_long = np.reshape(img, (-1, img.shape[-1]))

pca_obj = PCA(2)
pca_img_long = pca_obj.fit_transform(img_long)

min_x, max_x = min(pca_img_long[:,0]), max(pca_img_long[:,0])
min_y, max_y = min(pca_img_long[:,1]), max(pca_img_long[:,1])

x = np.linspace(min_x, max_x, 10)
y = np.linspace(min_y, max_y, 10)
mesh = np.meshgrid(x,y) 
mesh_colors = np.zeros((*mesh[0].shape, 3))
min_ind_array = np.zeros(mesh[0].shape)
for i in trange(len(x)):
    for j in range(len(y)):
        x_val = mesh[0][i,j]
        y_val = mesh[1][i,j]
        min_ind = np.argmin(np.linalg.norm(pca_img_long - np.array(x_val, y_val),axis=-1))
        mesh_colors[i,j] = img_long[min_ind]
        min_ind_array[i,j] = min_ind

wanted_point = np.array([0, -0.5])
dists = np.linalg.norm(pca_img_long - wanted_point,axis=-1)
pca_img_long[np.argmin(dists)]

# plt.imshow(mesh[0])
# plt.show()

fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
thinning = 500
ax[0,0].scatter(
        *pca_img_long[::thinning].T, 
        c = img_long[::thinning],
        edgecolor = 'k',
        # alpha = 0.1
        )
im = ax[0,1].hist2d(*pca_img_long.T, bins = 50, norm = LogNorm())
ax[1,0].pcolormesh(x, y, mesh_colors)
for i in trange(len(x)):
    for j in range(len(y)):
        x_val = mesh[0][i,j]
        y_val = mesh[1][i,j]
        ax[1,0].scatter(x_val, y_val, color = 'k')
        min_ind = int(min_ind_array[i,j])
        min_vec = pca_img_long[min_ind]
        ax[1,0].plot([x_val, min_vec[0]], [y_val, min_vec[1]], '-x', color = 'r')
        ax[1,0].text(x_val, y_val, s = f'({i,j})')
plt.show()

# Perform k-means with k = 3
# clust = KMeans(3)
clust = GaussianMixture(3)
clust.fit(img_long[::100])
labels = clust.predict(img_long)
labels_square = np.reshape(labels, img.shape[:2])

# Get mean values of each cluster
unique_labels = np.unique(labels)
mean_clust = []
for this_label in unique_labels:
    label_inds = labels == this_label
    mean_clust.append(img_long[label_inds].mean(axis=0))

ax = plt.axes(projection='3d')
thinning = 200
ax.scatter(*img_long[::thinning].T, 
           # c = labels[::thinning], 
           c = img_long[::thinning],
           alpha = 0.1,
           cmap = 'tab10',
           edgecolor = 'k')
for this_mean in mean_clust:
    ax.scatter(*this_mean, c = 'red', s = 200)
plt.show()

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].imshow(img)
ax[1].imshow(labels_square)
plt.show()

##############################
# Pass through an autoencoder to reduce dims



# Import common packages
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define networks
# import torch
# import torch.nn as nn
# from torch.nn import init
# from torch.nn import functional as F
# import math
#
# class autoencoderRNN(nn.Module):
#     """
#     Input and output transformations are encoder and decoder architectures
#     RNN will learn dynamics of latent space
#
#     Output has to be rectified
#     Can add dropout to RNN and autoencoder layers
#     """
#     def __init__(
#             self, 
#             input_size, 
#             hidden_size,  
#             output_size, 
#             dropout = 0.2,
#             ):
#         """
#         3 sigmoid layers for input and output each, to project between:
#             encoder : input -> latent
#             rnn : latent -> latent
#             decoder : latent -> output
#         """
#         super(autoencoderRNN, self).__init__()
#         self.encoder = nn.Sequential(
#                 nn.Linear(input_size, sum((input_size, hidden_size))//2),
#                 nn.Sigmoid(),
#                 nn.Linear(sum((input_size, hidden_size))//2, hidden_size),
#                 nn.Sigmoid(),
#                 )
#         self.bottleneck = nn.Sequential(
#                 nn.Linear(hidden_size),
#                 nn.Sigmoid()
#                 )
#         # self.rnn = nn.RNN(
#         #         hidden_size, 
#         #         hidden_size, 
#         #         rnn_layers, 
#         #         batch_first=False, 
#         #         bidirectional=False,
#         #         dropout = dropout,
#         #         )
#         self.decoder = nn.Sequential(
#                 nn.Linear(hidden_size, sum((hidden_size, output_size))//2),
#                 nn.Sigmoid(),
#                 nn.Linear(sum((hidden_size, output_size))//2, output_size),
#                 )
#         self.en_dropout = nn.Dropout(p = dropout)
#
#     def forward(self, x):
#         out = self.encoder(x)
#         out = self.en_dropout(out)
#         # latent_out, _ = self.rnn(out)
#         latent_out = self.bottleneck(out)
#         out = self.decoder(latent_out)
#         return out, latent_out
#

############################################################
img_dir = os.path.join(data_dir, 'balanced_images') 
img_path_list = sorted(glob(os.path.join(img_dir, '*')))

img_list = [plt.imread(x)[...,:3] for x in img_path_list]

# Downsample img
down_rate = 4
img_list = [x[::down_rate] for x in img_list]
img_list = [x[:,::down_rate] for x in img_list]

flat_img_list = [x.reshape(-1, 3) for x in img_list]

img_long = np.concatenate(flat_img_list)

pca_obj = PCA(2)
pca_img_long = pca_obj.fit_transform(img_long)

# clust = KMeans(3)
clust = GaussianMixture(4)
clust.fit(img_long[::100])
labels = clust.predict(img_long)

# Get mean values of each cluster
unique_labels = np.unique(labels)
mean_clust = []
for this_label in unique_labels:
    label_inds = labels == this_label
    mean_clust.append(pca_img_long[label_inds].mean(axis=0))


fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
thinning = 1000
ax[0,0].scatter(
        *pca_img_long[::thinning].T, 
        c = img_long[::thinning],
        )
im = ax[0,1].hist2d(*pca_img_long[::thinning].T, bins = 50, norm = LogNorm())
ax[1,0].scatter(
        *pca_img_long[::thinning].T, 
        c = labels[::thinning], 
        alpha = 0.1,
        zorder = 0,
        )
for this_mean in mean_clust:
    ax[1,0].scatter(*this_mean, c = 'red', s = 200, zorder = 10)
plt.show()

# Plot
out_dir = os.path.join(img_dir, 'labelled')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, this_img in enumerate(img_list):
    flat_img = this_img.reshape(-1, 3)
    flat_pred = clust.predict(flat_img)
    sq_pred = flat_pred.reshape(this_img.shape[:2])
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].imshow(this_img)
    ax[1].imshow(sq_pred)
    # plt.show()
    fig.savefig(os.path.join(out_dir, f'img_{i}.png'),
                bbox_inches='tight', dpi = 300)
    plt.close(fig)
