import pandas as pd
from tqdm import tqdm, trange
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from src import config

# Get directories from config
data_dir = config['data_dir']
export_json_path = os.path.join(
    # data_dir, #'labelled_images/ndjson/Export_project-trichrome_vessels_6_25-7_30_2025.ndjson')
    # "/home/abuzarmahmood/projects/autoslide_analysis/data/double_annotation_export.ndjson")
    "/home/abuzarmahmood/projects/autoslide_data/labelled_images/ndjson/Export_project-trichrome_vessels_6_25-9_15_2025.ndjson")
# data_dir, 'labelled_images/ndjson/Export_project-trichrome_vessels_6_25-6_27_2025.ndjson')

print(f"Loading export data from: {export_json_path}")
if not os.path.exists(export_json_path):
    print(f"ERROR: Export file not found at {export_json_path}")
    exit(1)

export_df = pd.read_json(export_json_path, lines=True)
print(f"Loaded {len(export_df)} rows from export file")
# export_df = pd.DataFrame(export_df['data_row'].values.tolist())

filename_list = []
polygon_list = []
object_num_list = []
all_filenames = set()

print("Parsing annotations from export data...")
for row_ind in trange(len(export_df)):
    # this_row = export_df.iloc[0]
    this_row = export_df.iloc[row_ind]
    file_name = this_row['data_row']['external_id']
    all_filenames.add(file_name)
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

print(f"Found {len(polygon_df)} polygon annotations across {len(all_filenames)} unique files")

# Find filepaths and plot overlays
filenames = list(all_filenames)
path_list = []
print("Searching for image files in data directory...")
missing_files = []
for this_name in filenames:
    # Search for this_name in data_dir
    basename = os.path.basename(this_name)
    glob_str = os.path.join(data_dir, '**', basename)
    filepaths = glob(glob_str, recursive=True)
    if len(filepaths) == 0:
        path_list.append(None)
        missing_files.append(this_name)
    else:
        path_list.append(filepaths[0])

if missing_files:
    print(f"WARNING: Could not find {len(missing_files)} files in data directory:")
    for missing_file in missing_files[:5]:  # Show first 5 missing files
        print(f"  - {missing_file}")
    if len(missing_files) > 5:
        print(f"  ... and {len(missing_files) - 5} more")

path_map = dict(zip(filenames, path_list))
polygon_df['filepath'] = polygon_df['filename'].map(path_map)

rows_before = len(polygon_df)
polygon_df = polygon_df.dropna()
rows_after = len(polygon_df)
if rows_before != rows_after:
    print(f"Dropped {rows_before - rows_after} rows due to missing image files")
print(f"Processing {rows_after} annotations with valid image paths")

# Copy all images to a directory
copy_dir = os.path.join(data_dir, 'labelled_images', 'images')
os.makedirs(copy_dir, exist_ok=True)
print(f"Copying images to: {copy_dir}")
copied_count = 0
for this_name, this_path in path_map.items():
    if this_path is not None:
        try:
            os.system(f'cp {this_path} {copy_dir}')
            copied_count += 1
        except Exception as e:
            print(f"ERROR copying {this_path}: {e}")
print(f"Copied {copied_count} images")

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
print(f"Creating masks in: {mask_dir}")
print(f"Creating test plots in: {test_plot_dir}")

# Process all files found in ndjson, including those without annotations
print("Processing images and creating masks...")
processed_count = 0
error_count = 0
for filename in tqdm(filenames):
    if filename not in path_map or path_map[filename] is None:
        continue

    this_filepath = path_map[filename]
    this_filename = os.path.basename(filename)
    filename_stem = os.path.splitext(this_filename)[0]

    # Get polygons for this file (if any)
    file_polygons = polygon_df[polygon_df['filename'] == filename]

    try:
        this_img = plt.imread(this_filepath)
        width = this_img.shape[1]
        height = this_img.shape[0]
    except Exception as e:
        print(f"ERROR reading image {this_filepath}: {e}")
        error_count += 1
        continue

    img_list = []
    if len(file_polygons) > 0:
        # File has annotations - create masks from polygons
        for this_row in file_polygons.iterrows():
            this_polygon = this_row[1]['polygon']
            this_obj_num = this_row[1]['object_num']
            poly_x = [x['x'] for x in this_polygon]
            poly_y = [x['y'] for x in this_polygon]

            polygon = [(x, y) for x, y in zip(poly_x, poly_y)]
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(
                polygon, outline=this_obj_num+1, fill=this_obj_num+1)
            img_list.append(img)

        summed_img = np.sum(np.array(img_list), axis=0)
        summed_img = summed_img / np.max(summed_img)
    else:
        # File has no annotations - create empty mask
        summed_img = np.zeros((height, width))

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
    try:
        Image.fromarray(mask).save(mask_filepath)
        processed_count += 1
    except Exception as e:
        print(f"ERROR saving mask {mask_filepath}: {e}")
        error_count += 1

print(f"Successfully processed {processed_count} images")
if error_count > 0:
    print(f"Encountered {error_count} errors during processing")

############################################################
# Validation step
############################################################
# Many of the masks did not line up with the images.
# If a name cannot be found in the test_plots dir, delete the mask and image

print("\nStarting validation step...")
val_img_paths = glob(os.path.join(test_plot_dir, '*.png'))
val_basenames = [os.path.basename(x) for x in val_img_paths]
print(f"Found {len(val_basenames)} validation plots")

wanted_mask_names = [x.replace('.png', '_mask.png') for x in val_basenames]
mask_paths = glob(os.path.join(mask_dir, '*.png'))
img_paths = glob(os.path.join(copy_dir, '*.png'))
print(f"Found {len(mask_paths)} masks and {len(img_paths)} images to validate")

print("Cleaning up mismatched masks...")
del_mask_count = 0
for this_mask_path in mask_paths:
    this_mask_name = os.path.basename(this_mask_path)
    if this_mask_name not in wanted_mask_names:
        try:
            os.remove(this_mask_path)
            del_mask_count += 1
        except Exception as e:
            print(f'ERROR deleting mask {this_mask_path}: {e}')

print("Cleaning up mismatched images...")
del_img_count = 0
for this_img_path in img_paths:
    this_img_name = os.path.basename(this_img_path)
    if this_img_name not in val_basenames:
        try:
            os.remove(this_img_path)
            del_img_count += 1
        except Exception as e:
            print(f'ERROR deleting image {this_img_path}: {e}')

print(f"\nValidation complete:")
print(f"  - Deleted {del_mask_count} mismatched masks")
print(f"  - Deleted {del_img_count} mismatched images")
print(f"  - Final dataset: {len(val_basenames)} image-mask pairs")
