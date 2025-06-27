"""
Take annotated csv and images
Merge marked tissues and label appropriately with tissue type
"""

import pylab as plt
import numpy as np 
import os
from pprint import pprint
from glob import glob
import pandas as pd
from tqdm import tqdm
from skimage.color import label2rgb
from ast import literal_eval

# Define project directory
auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide'

data_dir = os.path.join(auto_slide_dir, 'data')
init_annot_dir = os.path.join(data_dir, 'initial_annotation')
file_list = os.listdir(init_annot_dir)

fin_annotation_dir = os.path.join(data_dir, 'final_annotation')
if not os.path.exists(fin_annotation_dir):
    os.makedirs(fin_annotation_dir)

basenames = [os.path.basename(x).split('.')[0] for x in file_list] 
unique_basenames = np.unique(basenames)

############################################################

for this_basename in tqdm(unique_basenames):
    # this_basename = unique_basenames[0]

    metadata = pd.read_csv(os.path.join(init_annot_dir, this_basename + '.csv'))

    # Make sure tissue_num >0 as 0 is background
    assert np.all(metadata['tissue_num'] > 0), 'tissue_num should be >0'

    mask = np.load(os.path.join(init_annot_dir, this_basename + '.npy'))

    label_map = {}
    for i, row in metadata.iterrows():
        label_map[row['label']] = row['tissue_num']

    # Also get string to label each tissue
    metadata['tissue_str'] = metadata['tissue_num'].astype(str) + '_' + metadata['tissue_type']

    # Map values in mask according to label_map
    for key, value in label_map.items():
        mask[mask == key] = value

    # Plot mask and also write to file
    image_label_overlay = label2rgb(mask, 
                                    image=mask>0, 
                                    bg_label=0)

    fig, ax = plt.subplots(figsize = (5, 10))
    ax.imshow(image_label_overlay, cmap = 'tab10')
    ax.set_title(this_basename)
    # Label with tissue type
    for i, row in metadata.iterrows():
        centroid = literal_eval(row['centroid'])
        ax.text(centroid[1], centroid[0],
                row['tissue_str'], color = 'red',
                fontsize = 25, weight = 'bold')
    fig.savefig(os.path.join(fin_annotation_dir, this_basename + '.png'))
    plt.close(fig)

    np.save(os.path.join(fin_annotation_dir, this_basename + '.npy'), mask)

