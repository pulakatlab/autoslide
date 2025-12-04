import os
import json


def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    # Default data directory: project_root/test_data
    # Navigate from autoslide/src/ to project root, then to test_data
    default_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'test_data'
    )
    default_data_dir = os.path.abspath(default_data_dir)
    
    # Allow environment variable override
    data_dir = os.environ.get('AUTOSLIDE_DATA_DIR')
    
    if not data_dir and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            path_dict = json.load(f)
            config_data_dir = path_dict.get('data_dir', '')
            # If config path is relative, make it absolute relative to project root
            if config_data_dir and not os.path.isabs(config_data_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                data_dir = os.path.abspath(os.path.join(project_root, config_data_dir))
            elif config_data_dir:
                data_dir = config_data_dir
    
    if not data_dir:
        data_dir = default_data_dir

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
