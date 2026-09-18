"""
Upload a stratified random sample of new (unlabeled) heart-section images to a
Labelbox dataset.

Candidates are images in `heart_sections/` that trace back to a row in some
`*_section_frame.csv` (via get_section_from_hash.load_tracking_data /
load_suggested_regions) and whose hash is not already present in
`labelled_images/images/`. A sample is drawn, stratified proportionally by
slide, and uploaded with a de-dup check against Labelbox itself.
"""

import argparse
import csv
import datetime
import os

import labelbox as lb
import numpy as np

from autoslide.src import config
from autoslide.src.utils.get_section_from_hash import (
    load_tracking_data,
    load_suggested_regions,
)

TARGET_DATASET_ID = "cmtlv1vkb004a0752db3p58si"
SAMPLE_SIZE = 200
RANDOM_SEED = 42

HEART_SECTIONS_DIR = os.path.join(
    config["suggested_regions_dir"], "heart_sections")
LABELLED_IMAGES_DIR = os.path.join(config["labelled_data_dir"], "images")
UPLOAD_LOG_PATH = os.path.join(
    config["suggested_regions_dir"], "heart_sections_labelbox_upload_log.csv"
)


def hash_from_filename(filename):
    return filename.split("_")[-1].split(".")[0]


def build_candidate_table():
    """Return a DataFrame of not-yet-labeled candidate hashes, one row per
    hash, with columns: hash, slide, file_path."""
    paths, basenames, data_paths, scene_indices = load_tracking_data(
        config["tracking_dir"])
    final_df = load_suggested_regions(
        paths, basenames, data_paths, scene_indices)

    heart_files = os.listdir(HEART_SECTIONS_DIR)
    heart_hashes = {hash_from_filename(f): f for f in heart_files}

    known_hashes = set(final_df["section_hash"].astype(str))
    untraceable = set(heart_hashes) - known_hashes
    if untraceable:
        print(f"Warning: {len(untraceable)} heart_sections hashes are not "
              f"traceable to any section_frame row; dropping them.")
        for h in untraceable:
            heart_hashes.pop(h, None)

    labeled_hashes = {hash_from_filename(f)
                      for f in os.listdir(LABELLED_IMAGES_DIR)}

    candidate_hashes = set(heart_hashes) - labeled_hashes

    slide_by_hash = dict(zip(final_df["section_hash"].astype(str),
                             final_df["basename"]))

    rows = [
        {
            "hash": h,
            "slide": slide_by_hash[h],
            "file_path": os.path.join(HEART_SECTIONS_DIR, heart_hashes[h]),
        }
        for h in candidate_hashes
    ]
    return rows


def stratified_sample(rows, sample_size, seed):
    """Sample `sample_size` rows, allocated proportionally per slide (largest
    remainder rounding), sampled without replacement within each slide."""
    rng = np.random.default_rng(seed)

    by_slide = {}
    for row in rows:
        by_slide.setdefault(row["slide"], []).append(row)

    total = len(rows)
    raw_alloc = {slide: len(group) / total * sample_size
                 for slide, group in by_slide.items()}
    base_alloc = {slide: int(np.floor(n)) for slide, n in raw_alloc.items()}
    remainder = sample_size - sum(base_alloc.values())

    remainders = sorted(
        by_slide.keys(),
        key=lambda s: raw_alloc[s] - base_alloc[s],
        reverse=True,
    )
    for slide in remainders[:remainder]:
        base_alloc[slide] += 1

    sampled = []
    for slide, n in base_alloc.items():
        group = by_slide[slide]
        n = min(n, len(group))
        idx = rng.choice(len(group), size=n, replace=False)
        sampled.extend(group[i] for i in idx)
        print(f"  {slide}: sampled {n} / {len(group)}")

    return sampled


def filter_existing_in_labelbox(client, rows):
    """Drop any rows whose global_key already exists somewhere in Labelbox."""
    hashes = [row["hash"] for row in rows]
    result = client.get_data_row_ids_for_global_keys(hashes)
    existing = {h for h, data_row_id in zip(
        hashes, result["results"]) if data_row_id}
    if existing:
        print(f"Dropping {len(existing)} candidates that already exist in "
              f"Labelbox (by global_key).")
    return [row for row in rows if row["hash"] not in existing]


def upload(client, dataset_id, rows):
    dataset = client.get_dataset(dataset_id)
    assets = [
        {
            "row_data": row["file_path"],
            "global_key": row["hash"],
            "media_type": "IMAGE",
        }
        for row in rows
    ]
    task = dataset.create_data_rows(assets)
    task.wait_till_done()
    print("Upload errors:", task.errors)
    return task


def write_upload_log(rows, dataset_id):
    file_exists = os.path.exists(UPLOAD_LOG_PATH)
    uploaded_at = datetime.datetime.now().isoformat()
    with open(UPLOAD_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["hash", "slide", "file_path", "dataset_id", "uploaded_at"])
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({**row, "dataset_id": dataset_id,
                             "uploaded_at": uploaded_at})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the sample without uploading.")
    args = parser.parse_args()

    candidates = build_candidate_table()
    print(f"Total not-yet-labeled candidates: {len(candidates)}")

    print("Stratified sample allocation:")
    sample = stratified_sample(candidates, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Total sampled: {len(sample)}")

    if args.dry_run:
        for row in sample[:5]:
            print(row)
        return

    api_key = os.getenv("LABELBOX_API_KEY")
    if not api_key:
        raise RuntimeError("LABELBOX_API_KEY environment variable is not set.")
    client = lb.Client(api_key)

    sample = filter_existing_in_labelbox(client, sample)
    upload(client, TARGET_DATASET_ID, sample)
    write_upload_log(sample, TARGET_DATASET_ID)


if __name__ == "__main__":
    main()
