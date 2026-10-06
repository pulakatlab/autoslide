"""
Parse a Labelbox ndjson export into training images and instance masks.

For every row of the export this finds the image, rasterises the polygons of ALL
annotators of that image into one mask, and writes

    <data_dir>/labelled_images/images/<name>.png
    <data_dir>/labelled_images/masks/<name>_mask.png
    <data_dir>/labelled_images/test_plots/<name>.png     (image | mask QC plot)

Mask encoding (union of annotators)
-----------------------------------
The polygons of all annotators are OR-ed into one binary mask and its connected
components become the instances.  A component with id ``i`` (1..k) is stored with
gray value ``255 - (i - 1) * (127 // k)``.  This keeps every instance distinct
(``AugmentedCustDat`` treats each distinct gray value as an instance) and every
instance above 0.5 once scaled to [0, 1] (the evaluation code binarises masks at
0.5), so ``mask > 0`` is exactly the union of all polygons.  Overlapping polygons
from different annotators (or from one annotator) merge into one instance.
Images with no polygons get an all-zero mask (a skipped Labelbox label is a
negative image).

Usage:
    python -m autoslide.src.pipeline.label_handling.parse_exported_labels \
        [--ndjson PATH] [--data-dir DIR] [--extra-image-dir DIR ...]
"""

import argparse
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import ndimage  # noqa: E402
from tqdm import tqdm, trange  # noqa: E402

DEFAULT_NDJSON = os.path.join(
    '/home/abuzarmahmood/projects/autoslide_data/labelled_images/ndjson',
    'label_annotations_final_with_danyal.ndjson')

# Directories under data_dir that never hold source ROI images
_SKIP_DIRS = {'svs', 'fibrosis_results', 'tracking', '_archive', 'artifacts', 'plots',
              'masks', 'test_plots', 'augmented_images', 'augmented_masks', 'ndjson',
              'overlays', 'initial_annotation', 'final_annotation', '.dvc', '.git'}

MAX_INSTANCES = 127


def instance_values(k):
    """Gray values (descending from 255) used for instances 1..k."""
    assert 1 <= k <= MAX_INSTANCES, f'{k} instances cannot be encoded'
    step = 127 // k
    return [255 - i * step for i in range(k)]


def build_instance_mask(polygons, width, height):
    """
    Union-of-polygons instance mask.

    Args:
        polygons: iterable of polygons, each a list of (x, y) tuples
        width, height: mask size in pixels
    Returns:
        (mask, n_instances): uint8 array (height, width), number of connected components
    """
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([tuple(p) for p in poly], outline=1, fill=1)
    binary = np.array(canvas) > 0
    labelled, k = ndimage.label(binary)  # 4-connectivity
    mask = np.zeros((height, width), dtype=np.uint8)
    if k:
        lut = np.zeros(k + 1, dtype=np.uint8)
        lut[1:] = instance_values(k)
        mask = lut[labelled]
    return mask, k


def read_polygons(ndjson_path):
    """
    Returns:
        filenames: list of external ids (one per row)
        polygons: dict external_id -> list of polygons [(x, y), ...] (all annotators)
    """
    export_df = pd.read_json(ndjson_path, lines=True)
    print(f'Loaded {len(export_df)} rows from {ndjson_path}')
    filenames, polygons = [], {}
    for row_ind in trange(len(export_df), desc='Parsing annotations'):
        row = export_df.iloc[row_ind]
        name = row['data_row']['external_id']
        filenames.append(name)
        polys = polygons.setdefault(name, [])
        for project in row['projects'].values():
            for label in project['labels']:
                for obj in label['annotations']['objects']:
                    if 'polygon' in obj:
                        polys.append([(p['x'], p['y'])
                                     for p in obj['polygon']])
    return filenames, polygons


def index_images(data_dir, extra_image_dirs=()):
    """basename -> path for every PNG under data_dir outside the excluded dirs
    (one walk instead of one recursive glob per image) plus any extra_image_dirs
    (e.g. a backup of a previous images/ dir that holds ROIs no longer under
    data_dir). Earlier match wins: extra dirs / labelled_images/images, then
    review_image_cache, then everything else."""
    extra = [os.path.abspath(d) for d in extra_image_dirs]

    def priority(path):
        if any(os.path.abspath(path).startswith(d + os.sep) for d in extra):
            return 0
        if f'{os.sep}labelled_images{os.sep}images{os.sep}' in path:
            return 0
        if 'review_image_cache' in path:
            return 1
        return 2
    index = {}
    for d in extra:
        for f in os.listdir(d):
            if f.endswith('.png'):
                index[f] = os.path.join(d, f)
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            if f.endswith('.png'):
                path = os.path.join(root, f)
                if f not in index or priority(path) < priority(index[f]):
                    index[f] = path
    return index


def main(ndjson_path=DEFAULT_NDJSON, data_dir=None, extra_image_dirs=()):
    if data_dir is None:
        from autoslide.src import config
        data_dir = config['data_dir']
    if not os.path.exists(ndjson_path):
        raise SystemExit(f'ERROR: export file not found at {ndjson_path}')

    filenames, polygons = read_polygons(ndjson_path)
    all_names = list(dict.fromkeys(filenames))

    print('Indexing image files...')
    index = index_images(data_dir, extra_image_dirs)
    path_map = {n: index.get(os.path.basename(n)) for n in all_names}
    missing = [n for n, p in path_map.items() if p is None]
    if missing:
        print(
            f'WARNING: {len(missing)} files not found under {data_dir}; first: {missing[:5]}')

    img_dir = os.path.join(data_dir, 'labelled_images', 'images')
    mask_dir = os.path.join(data_dir, 'labelled_images', 'masks')
    plot_dir = os.path.join(data_dir, 'labelled_images', 'test_plots')
    for d in (img_dir, mask_dir, plot_dir):
        os.makedirs(d, exist_ok=True)

    processed, errors = 0, 0
    n_pos = n_neg = 0
    for name in tqdm(all_names, desc='Writing images and masks'):
        src = path_map[name]
        if src is None:
            continue
        base = os.path.basename(name)
        stem = os.path.splitext(base)[0]
        try:
            img = Image.open(src)
            width, height = img.size
            mask, k = build_instance_mask(polygons[name], width, height)
            dst = os.path.join(img_dir, base)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            Image.fromarray(mask).save(
                os.path.join(mask_dir, stem + '_mask.png'))
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img)
            ax[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
            fig.savefig(os.path.join(plot_dir, stem + '.png'))
            plt.close(fig)
            processed += 1
            n_pos += k > 0
            n_neg += k == 0
        except Exception as e:  # noqa: BLE001
            print(f'ERROR processing {src}: {e}')
            errors += 1

    print(
        f'Processed {processed} images ({n_pos} with vessels, {n_neg} without); {errors} errors')

    # Validation: keep only image/mask pairs that have a QC plot
    plot_names = {p for p in os.listdir(plot_dir) if p.endswith('.png')}
    wanted_masks = {p.replace('.png', '_mask.png') for p in plot_names}
    for d, wanted in ((mask_dir, wanted_masks), (img_dir, plot_names)):
        for f in os.listdir(d):
            if f.endswith('.png') and f not in wanted:
                os.remove(os.path.join(d, f))
    print(
        f'Final dataset: {len(os.listdir(img_dir))} images, {len(os.listdir(mask_dir))} masks')
    return processed


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ndjson', default=DEFAULT_NDJSON)
    ap.add_argument('--data-dir', default=None,
                    help='default: autoslide config data_dir')
    ap.add_argument('--extra-image-dir', action='append', default=[],
                    help='additional directory of source PNGs (repeatable)')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.ndjson, args.data_dir, args.extra_image_dir)
