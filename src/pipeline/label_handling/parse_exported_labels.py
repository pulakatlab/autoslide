import pandas as pd
from tqdm import tqdm, trange
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

export_json_path = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/Export_catalog_query_3_10_2025.ndjson'
# export_json_path = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/Export_catalog_query_3_10_2025_1_img.ndjson'

export_df = pd.read_json(export_json_path, lines=True)
# export_df = pd.DataFrame(export_df['data_row'].values.tolist())

filename_list = []
polygon_list = []
for row_ind in trange(len(export_df)):
    # this_row = export_df.iloc[0]
    this_row = export_df.iloc[row_ind]
    file_name = this_row['data_row']['external_id']
    projects = this_row['projects']
    project_keys = list(projects.keys())

    # this_key = project_keys[0]
    for this_key in project_keys:
        labels = export_df.iloc[0]['projects'][this_key]['labels']

        for this_label in labels:
            objects = this_label['annotations']['objects']
            for this_object in objects:
                polygon = this_object['polygon']
                filename_list.append(file_name)
                polygon_list.append(polygon)

polygon_df = pd.DataFrame({'filename':filename_list, 'polygon':polygon_list})

data_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data'

# Find filepaths and plot overlays
filenames = polygon_df['filename'].unique()
path_list = []
for this_name in filenames:
    # Search for this_name in data_dir 
    glob_str = os.path.join(data_dir, '**', this_name)
    filepaths = glob(glob_str, recursive=True)[0]
    path_list.append(filepaths)

path_map = dict(zip(filenames, path_list))
polygon_df['filepath'] = polygon_df['filename'].map(path_map)

# Copy all images to a directory 
copy_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/images'
os.makedirs(copy_dir, exist_ok=True)
for this_name, this_path in path_map.items():
    os.system(f'cp {this_path} {copy_dir}')

# # Plot overlay
# ind = 0
# this_row = polygon_df.iloc[ind]
# this_filepath = this_row['filepath']
# this_polygon = this_row['polygon']
# this_img = plt.imread(this_filepath)
# poly_x = [x['x'] for x in this_polygon]
# poly_y = [x['y'] for x in this_polygon]
# plt.imshow(this_img)
# plt.plot(poly_x, poly_y, '-o', c='y')
# plt.show()

# Create mask and save to dir
mask_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/masks'
os.makedirs(mask_dir, exist_ok=True)

for ind in trange(len(polygon_df)):
    this_row = polygon_df.iloc[ind]
    this_filepath = this_row['filepath']
    this_polygon = this_row['polygon']
    this_img = plt.imread(this_filepath)
    poly_x = [x['x'] for x in this_polygon]
    poly_y = [x['y'] for x in this_polygon]

    # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    # width = ?
    # height = ?
    width = this_img.shape[1]
    height = this_img.shape[0]
    polygon = [(x,y) for x,y in zip(poly_x, poly_y)]
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)*255

    mask_filename = os.path.basename(this_filepath).replace('.png', '_mask.png')
    mask_filepath = os.path.join(mask_dir, mask_filename)
    Image.fromarray(mask).save(mask_filepath)


# width = this_img.shape[1]
# height = this_img.shape[0]
# polygon = [(x,y) for x,y in zip(poly_x, poly_y)]
# img = Image.new('L', (width, height), 0)
# ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
# mask = np.array(img)
#
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(this_img)
# ax[1].imshow(mask)
# plt.show()
