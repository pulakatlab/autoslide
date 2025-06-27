"""
For prediction on every section, perform prediction on a radius larger than the image
and aggregate the predictions.
"""
import os
import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np 
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from importlib import reload
from autoslide import config

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = config['artifacts_dir']
plot_dir = config['plot_dirs']

# Import utilities directly
from autoslide.pipeline.utils import visualize_sections, get_section, gen_step_windows

# Create a utils namespace for backward compatibility
class Utils:
    visualize_sections = visualize_sections
    get_section = get_section
    gen_step_windows = gen_step_windows

utils = Utils()

##############################
mask_dir = os.path.join(data_dir, 'final_annotation') 
metadata_dir = os.path.join(data_dir, 'initial_annotation') 
section_dir = os.path.join(data_dir, 'suggested_regions')
data_path_list = glob(os.path.join(data_dir, '**', '*TRI*.svs'))

# Get paths to all metadata files
metadata_path_list = glob(os.path.join(section_dir, '**','*TRI*.csv'), recursive=True)
# mask_path_list = glob(os.path.join(mask_dir, '*TRI*.png'))
basenames = [os.path.basename(os.path.dirname(path)) for path in metadata_path_list]

basenames = sorted(basenames)
metadata_path_list = sorted(metadata_path_list)
# mask_path_list = sorted(mask_path_list)

matched_data_paths = [
        [x for x in data_path_list if basename in x.replace('-','_')][0] \
                for basename in basenames]

metadata_df = pd.DataFrame(
        data = {
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
fin_metadata_df = fin_metadata_df.drop_duplicates(subset = 'section_hash')

##############################
# Expand original section
##############################
og_image_path = os.path.join(
        data_dir,
        'labelled_images/images/1_heart_1022472553.png'
        )

# og_image_path = os.path.join(
#         data_dir,
#         'suggested_regions/TRI_130_163A_40490/images/4_heart_6988590045.png'
#         )
# sec_name = '4_heart_6988590045'
sec_name = os.path.basename(og_image_path).split('.')[0]
sec_hash = int(sec_name.split('_')[-1])

sec_metadata = fin_metadata_df[fin_metadata_df['section_hash'] == sec_hash]

data_path = sec_metadata.data_path.values[0]
slide = slideio.open_slide(data_path, 'SVS')
scene = slide.get_scene(0)

wanted_section = eval(sec_metadata.section_bounds.values[0])
# utils.visualize_sections(
#     scene, 
#     [wanted_section], 
#     )

# Get region around section
expand_ratio = 1.5
section_center = [int((wanted_section[0] + wanted_section[2])/2), int((wanted_section[1] + wanted_section[3])/2)] 
x_radius = int((wanted_section[2] - wanted_section[0])/2 * expand_ratio)
y_radius = int((wanted_section[3] - wanted_section[1])/2 * expand_ratio)

expanded_section = [
        section_center[0] - x_radius,
        section_center[1] - y_radius,
        section_center[0] + x_radius,
        section_center[1] + y_radius
        ]

expanded_section_size = (
        expanded_section[2] - expanded_section[0],
        expanded_section[3] - expanded_section[1]
        )

expanded_img = utils.get_section(scene, expanded_section, down_sample=10)

og_img = plt.imread(og_image_path)
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(og_img)
ax[1].imshow(expanded_img)
# Draw rectangle around wanted section
img_center = [int(expanded_img.shape[1]/2), int(expanded_img.shape[0]/2)]
x_radius = expanded_img.shape[0] / 2
y_radius = expanded_img.shape[1] / 2
adj_x_rad = x_radius / expand_ratio
adj_y_rad = y_radius / expand_ratio
ax[1].add_patch(plt.Rectangle(
    (img_center[0] - adj_x_rad, img_center[1] - adj_y_rad),
    2*adj_x_rad, 2*adj_y_rad,
    edgecolor='y',
    facecolor='none'
    ))
ax[0].set_title('Original Image')
ax[1].set_title('Expanded Image')
fig.suptitle(sec_name)
fig.savefig(os.path.join(plot_dir, f'{sec_name}_expansion_comparison.png'))
# plt.show()
plt.close(fig)

##############################
# Perform prediction by stepping through expanded section 
##############################

# step_list = gen_step_windows(
#         window_shape = np.array(expanded_img.shape[:2]) // 2,
#         image_shape = expanded_img.shape[:2],
#         overlap = 0.8,
#         )
# step_list = np.array(step_list)
#
step_list = utils.gen_step_windows(
        window_shape = np.array(expanded_section_size) // 2, 
        image_shape = expanded_section_size,
        overlap = 0.5,
        )
step_list = np.array(step_list)
step_list[:,0] = step_list[:,0] + expanded_section[0]
step_list[:,1] = step_list[:,1] + expanded_section[1]
step_list[:,2] = step_list[:,2] + expanded_section[0]
step_list[:,3] = step_list[:,3] + expanded_section[1]

from importlib import reload
reload(utils)

# Plot step_list
fig, ax = utils.visualize_sections(
        scene, step_list, return_image=True, crop_to_sections=True,
        down_sample=10,
        edgecolor=np.arange(len(step_list)),
        linewidth=5,
        )
fig.suptitle('Expanded Section Step List')
fig.savefig(os.path.join(plot_dir, f'{sec_name}_expanded_section_step_list.png'))
plt.close(fig)

##############################
# Load model
model_save_path = os.path.join(artifacts_dir, 'mask_rcnn_model.pth')

# Recrate base model
model = torchvision.models.detection.maskrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)

# Load model weights
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()


transform = T.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img = Image.open(og_image_path).convert("RGB")
ig = transform(expanded_img)
with torch.no_grad():
    pred = model([ig.to(device)])
n_preds = len(pred[0]["masks"])
all_preds = np.stack([(pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze() for i in range(n_preds)])
