import os
import json


def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            path_dict = json.load(f)
            data_dir = path_dict.get('data_dir', os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data'))
    else:
        data_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'data')

    return {
        'data_dir': data_dir,
        'svs_dir': os.path.join(data_dir, 'svs'),
        'artifacts_dir': os.path.join(data_dir, 'artifacts'),
        'plot_dirs': os.path.join(data_dir, 'plots'),
        'initial_annotation_dir': os.path.join(data_dir, 'initial_annotation'),
        'final_annotation_dir': os.path.join(data_dir, 'final_annotation'),
        'suggested_regions_dir': os.path.join(data_dir, 'suggested_regions'),
        'tracking_dir': os.path.join(data_dir, 'tracking'),
        'labelled_data_dir': os.path.join(data_dir, 'labelled_images'),
    }


# Load configuration when module is imported
config = load_config()
