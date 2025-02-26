"""
Sandbox for testing slideio
"""

import slideio
import pylab as plt
import cv2 as cv
import numpy as np 
import os
from pprint import pprint


data_dir = '/media/bigdata/projects/pulakat_lab/auto_slide/data/'
# data_path = os.path.join(data_dir, 'PSR 142B-155 146A-159 38740.svs')
data_path = os.path.join(data_dir, 'TRI 142B-155 146A-159 38717.svs')

slide = slideio.open_slide(data_path, 'SVS')
scene = slide.get_scene(0)
print(scene.name, scene.rect, scene.num_channels) 

metadata_str = slide.raw_metadata
pprint(metadata_str.split('|'))

# Get 'OriginalWidth' and 'OriginalHeight' from metadata
metadata = {}
for item in metadata_str.split('|'):
    key, value = item.split('=')
    metadata[key.strip()] = value.strip()

print(metadata['OriginalWidth'], metadata['OriginalHeight'])
og_width = int(metadata['OriginalWidth'])
og_height = int(metadata['OriginalHeight'])
magnification = int(metadata['AppMag'])

image = scene.read_block(size=(1000,0)) 

# Also read subsection of 10% heigth and width of image
sub_image = scene.read_block(
        rect=(0,0,int(0.1*og_width), int(0.1*og_height)),
        # size=(1000,0)
                             )
fig, ax = plt.subplots(2,1)
ax[0].imshow(image)
ax[0].axvline(int(image.shape[1]*0.1), color='r', linestyle='dashed', linewidth=2)
ax[0].axhline(int(image.shape[0]*0.1), color='r', linestyle='dashed', linewidth=2)
ax[1].imshow(sub_image)
plt.show()

plt.imshow(sub_image)
plt.show()

# Apply thresholding

# Convert to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
log_gray_image = np.log(gray_image)
# Rescale to 0-255
log_gray_image = log_gray_image - np.min(log_gray_image) 
log_gray_image = log_gray_image / np.max(log_gray_image)
log_gray_image = log_gray_image * 255
log_gray_image = log_gray_image.astype(np.uint8)

plt.imshow(log_gray_image)
plt.colorbar()
plt.show()

ret, thresh = cv.threshold(log_gray_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

plt.hist(gray_image.ravel(), bins=256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
# Set scaling to logarithmic
plt.yscale('log')
plt.show()

plt.imshow(thresh)
plt.colorbar()
plt.show()

# Show only masked region
masked = np.ma.masked_where(thresh == 255, gray_image) 
plt.imshow(masked, interpolation='nearest', aspect='auto')
plt.show()
