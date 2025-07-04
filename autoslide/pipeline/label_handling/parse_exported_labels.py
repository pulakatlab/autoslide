import pandas as pd
from tqdm import tqdm, trange
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from autoslide import config

# Get directories from config
data_dir = config['data_dir']
export_json_path = os.path.join(
    data_dir, 'labelled_images/ndjson/Export_project-trichrome_vessels_6_25-6_27_2025.ndjson')

export_df = pd.read_json(export_json_path, lines=True)
# export_df = pd.DataFrame(export_df['data_row'].values.tolist())

filename_list = []
polygon_list = []
object_num_list = []
for row_ind in trange(len(export_df)):
    # this_row = export_df.iloc[0]
    this_row = export_df.iloc[row_ind]
    file_name = this_row['data_row']['external_id']
    projects = this_row['projects']
    project_keys = list(projects.keys())

    # this_key = project_keys[0]
    for this_key in project_keys:
        labels = export_df.iloc[row_ind]['projects'][this_key]['labels']

        for this_label in labels:
            objects = this_label['annotations']['objects']
            for i, this_object in enumerate(objects):
                polygon = this_object['polygon']
                filename_list.append(file_name)
                polygon_list.append(polygon)
                object_num_list.append(i)

polygon_df = pd.DataFrame(
    {
        'filename': filename_list,
        'polygon': polygon_list,
        'object_num': object_num_list
    }
)

# Find filepaths and plot overlays
filenames = polygon_df['filename'].unique()
path_list = []
for this_name in filenames:
    # Search for this_name in data_dir
    basename = os.path.basename(this_name)
    glob_str = os.path.join(data_dir, '**', basename)
    filepaths = glob(glob_str, recursive=True)
    if len(filepaths) == 0:
        path_list.append(None)
    else:
        path_list.append(filepaths[0])

path_map = dict(zip(filenames, path_list))
polygon_df['filepath'] = polygon_df['filename'].map(path_map)

polygon_df = polygon_df.dropna()

# Copy all images to a directory
copy_dir = os.path.join(data_dir, 'labelled_images', 'images')
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
mask_dir = os.path.join(data_dir, 'labelled_images', 'masks')
test_plot_dir = os.path.join(data_dir, 'labelled_images', 'test_plots')
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(test_plot_dir, exist_ok=True)

polygon_groups = polygon_df.groupby('filename')
for ind, this_group in tqdm(polygon_groups):
    # this_row = polygon_df.iloc[ind]
    this_filepath = this_group.iloc[0]['filepath']
    this_filename = os.path.basename(this_group.iloc[0]['filename'])
    filename_stem = os.path.splitext(this_filename)[0]
    img_list = []
    for this_row in this_group.iterrows():
        this_polygon = this_row[1]['polygon']
        this_obj_num = this_row[1]['object_num']
        # this_polygon = this_row['polygon']
        this_img = plt.imread(this_filepath)
        poly_x = [x['x'] for x in this_polygon]
        poly_y = [x['y'] for x in this_polygon]

        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        # width = ?
        # height = ?
        width = this_img.shape[1]
        height = this_img.shape[0]
        polygon = [(x, y) for x, y in zip(poly_x, poly_y)]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(
            polygon, outline=this_obj_num+1, fill=this_obj_num+1)
        img_list.append(img)
    summed_img = np.sum(np.array(img_list), axis=0)
    summed_img = summed_img / np.max(summed_img)
    mask = np.array(summed_img)*255
    mask = mask.astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(this_img)
    ax[1].imshow(summed_img)
    # plt.show()
    fig.savefig(os.path.join(test_plot_dir, filename_stem + '.png'))
    plt.close(fig)

    mask_filename = os.path.basename(
        this_filepath).replace('.png', '_mask.png')
    mask_filepath = os.path.join(mask_dir, mask_filename)
    Image.fromarray(mask).save(mask_filepath)

############################################################
# Validation step
############################################################
# Many of the masks did not line up with the images.
# If a name cannot be found in the test_plots dir, delete the mask and image

val_img_paths = glob(os.path.join(test_plot_dir, '*.png'))
val_basenames = [os.path.basename(x) for x in val_img_paths]

wanted_mask_names = [x.replace('.png', '_mask.png') for x in val_basenames]
mask_paths = glob(os.path.join(mask_dir, '*.png'))
img_paths = glob(os.path.join(copy_dir, '*.png'))

del_mask_count = 0
for this_mask_path in mask_paths:
    this_mask_name = os.path.basename(this_mask_path)
    if this_mask_name not in wanted_mask_names:
        os.remove(this_mask_path)
        print(f'Deleted {this_mask_path}')
        del_mask_count += 1

del_img_count = 0
for this_img_path in img_paths:
    this_img_name = os.path.basename(this_img_path)
    if this_img_name not in val_basenames:
        os.remove(this_img_path)
        print(f'Deleted {this_img_path}')
        del_img_count += 1
