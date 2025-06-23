"""
Iterate through image given window size and stride, and return a list of
regions to be used for classification.
"""

auto_slide_dir = '/media/bigdata/projects/auto_slide'
# auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

import os
import sys
import uuid
sys.path.append(os.path.join(auto_slide_dir, 'src'))
from pipeline import utils

import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np 
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob
import json
from tqdm import tqdm
import hashlib

def remove_mask_edge(
        mask,
        mask_resolution,
        closing_len = 75e-6, # meters
        edge_len = 100e-6, # meters
        ):
    """
    Remove the edge of the mask to avoid getting sections on the edge

    Args:
        mask: np.array
        mask_resolution: float
        closing_len: float
        edge_len: float

    Returns:
        mask: np.array
    """
    dilate_kern_len = int(np.round((closing_len / mask_resolution))) 
    edge_kern_len = int(np.ceil((edge_len / mask_resolution)))
    dilate_kern = morph.disk(dilate_kern_len)
    erode_kern = morph.disk(edge_kern_len*2)

    dilated_mask = morph.binary_dilation(mask, dilate_kern)
    filled_mask = morph.erosion(dilated_mask, dilate_kern)
    eroded_mask = morph.erosion(filled_mask, erode_kern)

    return eroded_mask

def str_to_hash(s):
    """
    Convert a string to a hash value.
    
    Args:
        s (str): The input string.
    
    Returns:
        str: A hexadecimal hash value of the input string.
    """
    return hashlib.sha256(s.encode()).hexdigest()[:16]  # Shorten to 16 characters

# data_dir = '/media/storage/svs_tri_files'
# data_dir = '/media/fastdata/9_month_wistar_zdf_female/'
# data_dir = os.path.join(auto_slide_dir, 'data')
data_dir = os.path.join(auto_slide_dir, 'data')
tracking_dir = os.path.join(data_dir, '.tracking')

output_base_dir = os.path.join(data_dir, 'suggested_regions')
if not os.path.exists(output_base_dir):
    os.mkdir(output_base_dir)

# mask_dir = os.path.join(data_dir, 'final_annotation') 
# metadata_dir = os.path.join(data_dir, 'initial_annotation') 
file_list = os.listdir(tracking_dir)
json_path_list = glob(os.path.join(tracking_dir, '*.json'))
json_list = [json.load(open(x, 'r')) for x in json_path_list]

############################################################
# data_path_list = glob(os.path.join(data_dir, '*TRI*.svs'))

# data_path = os.path.join(data_dir, 'TRI 142B-155 146A-159 38717.svs')

# for data_path in data_path_list:
for this_json, json_path in tqdm(zip(json_list, json_path_list), total=len(json_list)): 
    data_path = this_json['data_path']
    try:
        # data_basename = os.path.basename(data_path).split('.')[0]
        data_basename = this_json['file_basename'].split('.')[0]
        # Replace spaces and dashes with underscores
        data_basename_proc = data_basename.replace(' ', '_').replace('-', '_')
        this_output_dir = os.path.join(output_base_dir, data_basename_proc) 

        if not os.path.exists(this_output_dir):
            os.mkdir(this_output_dir)

        # file_basename = os.path.basename(data_path).split('.')[0]
        # file_basename_proc = file_basename.replace(' ', '_')
        # label_mask_path = os.path.join(mask_dir, file_basename + '.npy')
        label_mask_path = this_json['fin_mask_path']
        label_mask = np.load(label_mask_path)
        # metadata_path = os.path.join(metadata_dir, file_basename + '.csv')
        metadata_path = this_json['wanted_regions_frame_path']
        metadata = pd.read_csv(metadata_path)

        mask = label_mask.copy()
        slide = slideio.open_slide(data_path, 'SVS')
        scene = slide.get_scene(0)
        resolution = scene.resolution[0] # meters / pixel
        size_meteres = np.array(scene.size) * np.array(resolution)
        down_mag = mask.shape[0] / scene.rect[2]
        mask_resolution = resolution / down_mag

        slide_metadata = utils.slide_handler(data_path)

        # Get tissue dimenions
        # wanted_labels = [1,4]
        wanted_labels = metadata.loc[metadata['tissue_type'] == 'heart']['tissue_num'].values 
        # wanted_mask = mask == wanted_label
        wanted_mask = np.isin(mask, wanted_labels) 

        # tissue_props = metadata.loc[metadata['tissue_num'] == wanted_label]
        # major_len = tissue_props['axis_major_length'].values[0]
        # minor_len = tissue_props['axis_minor_length'].values[0]
        # down_mag = mask.shape[0] / scene.rect[2]
        # len_array_raw = np.array([major_len, minor_len])
        # len_array_meters = (len_array_raw / down_mag) * resolution

        ##############################
        eroded_mask = remove_mask_edge(
                wanted_mask,
                mask_resolution,
                closing_len = 75e-6,
                edge_len = 100e-6,
                )


        window_len = 7e-4 # meters
        window_shape_pixels = int(window_len / resolution)

        window_shape = [window_shape_pixels, window_shape_pixels]
        step_shape = window_shape.copy()

        step_list = utils.gen_step_windows(step_shape, window_shape, scene.rect[2:])

        _,wanted_sections = utils.get_wanted_sections(
                scene,
                eroded_mask,
                step_list,
                min_fraction = 1,
                )

        fig, ax = utils.visualize_sections(
                scene,
                wanted_sections,
                plot_n = -1,
                edgecolor = 'orange',
                return_image = True,
                )
        ax.legend().set_visible(False)
        fig.savefig(
                os.path.join(
                    this_output_dir, 
                    # file_basename_proc + '_' + 'selected_section_visualization.png'),
                    data_basename_proc + '_' + 'selected_section_visualization.png'),
                dpi = 300,
                bbox_inches = 'tight',
                )
        plt.close(fig)

        fin_mask = mask.copy()
        fin_mask[~eroded_mask] = 0
        section_frame = utils.annotate_sections(
                scene,
                fin_mask, 
                metadata,
                wanted_sections,
                )

        section_labels = section_frame['label_values'].astype(str)+ '_' + \
            section_frame['tissue_type'] 

        section_frame['section_labels'] = section_labels

        # Generate truly unique identifiers for each section
        section_frame['section_hash'] = [
                # str(uuid.uuid4().int)[:16] \
                str_to_hash(data_basename_proc + '_' + str(section_frame.iloc[i])) \
                        for i in range(len(section_frame))
                        ]

        # Write out section_frame
        section_frame.to_csv(
                os.path.join(
                    this_output_dir,
                    # file_basename_proc + '_' + 'section_frame.csv'),
                    data_basename_proc + '_' + 'section_frame.csv'),
                index = False,
                )

        img_section_list, img_list = utils.output_sections(
                            scene,
                            section_frame['section_bounds'].to_list(),
                            this_output_dir,
                            down_sample = 4,
                            random_output_n = None,
                            output_type = 'return',
                            )

        # Write out images
        out_image_dir = os.path.join(this_output_dir, 'images')
        if not os.path.exists(out_image_dir):
            os.mkdir(out_image_dir)
        utils.write_out_images(
                img_list,
                section_frame,
                out_image_dir,
                )

        # Add path to section frame to json
        this_json['suggested_regions_frame_path'] = os.path.join(
                this_output_dir, 
                data_basename_proc + '_' + 'section_frame.csv')

        with open(json_path, 'w') as f:
            json.dump(this_json, f, indent=4)

    except Exception as e:
        print(e)
