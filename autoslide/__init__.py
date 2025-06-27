import os
import json

def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "data_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
            "auto_slide_dir": os.path.dirname(os.path.dirname(__file__)),
            "plot_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots'),
            "artifacts_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
        }

# Load configuration when module is imported
config = load_config()
