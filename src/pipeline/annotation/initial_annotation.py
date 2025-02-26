"""
Functions to help with annotation of functions

Preliminary idea being that tissues need to be manually delineated

Steps:
1. Load image
2. Output image
3. Image is manually annotated
4. Image is loaded back in
5. Image is convex hull'd and regions segmented
6. Separately, user is asked to input tissue type
"""
auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

import sys
sys.path.append(os.path.join(auto_slide_dir, 'src'))
import utils

import slideio
import pylab as plt
import cv2 as cv
import numpy as np 
import os
from pprint import pprint
from glob import glob
import pandas as pd
from tqdm import tqdm

from skimage.morphology import binary_dilation as dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters.rank import gradient
from scipy.ndimage import binary_fill_holes

############################################################
# PARAMS
down_sample = 100
dilation_kern_size = 2
area_threshold = 10000

############################################################


# data_dir = '/media/bigdata/projects/pulakat_lab/auto_slide/data/'
# data_dir = '/media/fastdata/9_month_wistar_zdf_female'
data_dir = os.path.join(auto_slide_dir, 'data')
glob_pattern = 'TRI*.svs'
file_list = glob(os.path.join(data_dir, glob_pattern))

annot_dir = os.path.join(data_dir, 'initial_annotation')
if not os.path.exists(annot_dir):
    os.makedirs(annot_dir)

for data_path in tqdm(file_list):
    # data_path = file_list[0]
    file_basename = os.path.basename(data_path)
    slide = slideio.open_slide(data_path, 'SVS')
    scene = slide.get_scene(0)
    image_rect = np.array(scene.rect) // down_sample
    image = scene.read_block(size = image_rect[2:])

    ############################################################

    threshold_mask = utils.get_threshold_mask(scene, down_sample = down_sample)

    dilation_kern = np.ones((dilation_kern_size, dilation_kern_size))
    dilated_mask = dilation(threshold_mask, footprint = dilation_kern) 

    label_image = label(dilated_mask)
    regions = regionprops(label_image)
    image_label_overlay = label2rgb(label_image, image=dilated_mask, bg_label=0)

    wanted_feature_names = [
            'label',
            'area',
            'eccentricity',
            'axis_major_length',
            'axis_minor_length',
            'eccentricity',
            'solidity',
            'centroid',
            ]

    wanted_features = [
            [getattr(region, feature_name) for region in regions]
            for feature_name in wanted_feature_names
            ]

    region_frame = pd.DataFrame(
            {feature_name: feature for feature_name, feature in \
                    zip(wanted_feature_names, wanted_features)}
            )

    region_frame.sort_values('label', ascending = True, inplace = True)
    wanted_regions_frame = region_frame[region_frame.area > area_threshold]
    wanted_regions = wanted_regions_frame.label.values

    # Output wanted_regions_frame to be annotated manually
    wanted_regions_frame['tissue_type'] = np.nan
    wanted_regions_frame['tissue_num'] = np.nan

    wanted_regions_frame.to_csv(
            os.path.join(annot_dir, file_basename.replace('.svs', '.csv')),
            index = False
            )

    # Drop regions that are not wanted
    fin_label_image = label_image.copy()
    for i in region_frame.label.values: 
        if i not in wanted_regions:
            fin_label_image[fin_label_image == i] = 0

    # Write out image with regions labelled
    np.save(
            os.path.join(annot_dir, file_basename.replace('.svs', '.npy')),
            fin_label_image
            )

    image_label_overlay = label2rgb(fin_label_image, 
                                    image=fin_label_image>0, 
                                    bg_label=0)


    filled_binary = binary_fill_holes(fin_label_image > 0)*1

    gradient_image = gradient(filled_binary, np.ones((10, 10)))
    grad_inds = np.where(gradient_image > 0)

    fig,ax = plt.subplots(1, 5, 
                          sharex = True, sharey = True,
                          figsize = (20, 10)
                          )
    ax[0].imshow(np.swapaxes(image, 0, 1))
    ax[1].imshow(threshold_mask)
    ax[2].imshow(dilated_mask)
    ax[3].imshow(image_label_overlay)
    ax[4].imshow(np.swapaxes(image, 0, 1))
    ax[4].scatter(grad_inds[1], grad_inds[0], 
                  s = 1, color = 'orange', alpha = 0.7,
                  label = 'Outline')
    ax[0].set_title('Original')
    ax[1].set_title('Threshold')
    ax[2].set_title('Dilated')
    ax[3].set_title('Labelled')
    ax[4].set_title('Outline Overlay')
    ax[4].legend()
    # Add labels at the center of each region
    for this_row in wanted_regions_frame.itertuples():
        ax[3].text(this_row.centroid[1], this_row.centroid[0], 
                   this_row.label, color = 'r', fontsize = 25,
                   weight = 'bold')
    fig.suptitle(file_basename)
    plt.tight_layout()
    fig.savefig(os.path.join(annot_dir, file_basename.replace('.svs', '.png')),
                bbox_inches = 'tight')
    plt.close(fig)
    # plt.show()
