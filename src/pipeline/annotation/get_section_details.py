"""
Given a section id, this script will return the details of the section from
the original data file.
"""

from src.pipeline import utils
import os
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob
from src import config

# Get directories from config
data_dir = config['data_dir']
plot_dir = config['plot_dirs']

# Import utilities directly

# Get directories from config
data_dir = config['data_dir']
mask_dir = config['final_annotation_dir']
metadata_dir = config['initial_annotation_dir']
section_dir = config['suggested_regions_dir']
data_path_list = glob(os.path.join(data_dir, '*TRI*.svs'))

# Get paths to all metadata files
metadata_path_list = glob(os.path.join(
    section_dir, '**', '*TRI*.csv'), recursive=True)
# mask_path_list = glob(os.path.join(mask_dir, '*TRI*.png'))
basenames = [os.path.basename(os.path.dirname(path))
             for path in metadata_path_list]

basenames = sorted(basenames)
metadata_path_list = sorted(metadata_path_list)
# mask_path_list = sorted(mask_path_list)

matched_data_paths = [
    [x for x in data_path_list if basename in x.replace('-', '_')][0]
    for basename in basenames]

metadata_df = pd.DataFrame(
    data={
        'basename': basenames,
        'metadata_path': metadata_path_list,
        'data_path': matched_data_paths,
        # 'mask_path': mask_path_list
    }
)

# Assert that each basename is in both metadata and mask paths
check_bool = []
for i, row in metadata_df.iterrows():
    check_bool.append(row['basename'] in row['metadata_path'])
    # check_bool.append(row['basename'] in row['mask_path'])

assert all(check_bool), "Error: basename not found in metadata or mask path"

# Load each metadata file and add basename to the dataframe
metadata_list = []
for i, row in metadata_df.iterrows():
    metadata = pd.read_csv(row['metadata_path'])
    metadata['basename'] = row['basename']
    metadata['data_path'] = row['data_path']
    metadata_list.append(metadata)

fin_metadata_df = pd.concat(metadata_list)
fin_metadata_df = fin_metadata_df.drop_duplicates(subset='section_hash')

##############################
og_image_path = '/media/storage/svs_tri_files/suggested_regions/TRI_130_163A_40490/images/4_heart_6988590045.png'
sec_name = '4_heart_6988590045'
sec_hash = int(sec_name.split('_')[-1])

sec_metadata = fin_metadata_df[fin_metadata_df['section_hash'] == sec_hash]

data_path = sec_metadata.data_path.values[0]
slide_handler = utils.slide_handler(data_path)
scene = slide_handler.scene

wanted_section = eval(sec_metadata.section_bounds.values[0])
# utils.visualize_sections(
#     scene,
#     [wanted_section],
#     )
img = utils.get_section(scene, wanted_section, down_sample=10)

og_img = plt.imread(og_image_path)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(og_img)
ax[1].imshow(img)
ax[0].set_title('Original Image')
ax[1].set_title('Extracted Image')
fig.suptitle(sec_name)
fig.savefig(os.path.join(plot_dir, f'{sec_name}_comparison.png'))
# plt.show()
plt.close(fig)
