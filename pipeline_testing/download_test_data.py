"""
Standalone script to download test data from Google Drive

This script downloads test data without running any pipeline steps.
"""

import os
import sys
import gdown

# Add project root to path
script_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, project_root)


def download_test_data():
    """Download test data from Google Drive"""
    print("Downloading test data from Google Drive...")
    
    # Google Drive folder ID
    folder_id = "165Ei63lVEtCI1aQpKYIqhNR_zE8YeG5c"
    
    # Create test data directory
    test_data_dir = os.path.join(project_root, 'test_data', 'svs')
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Download folder contents
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading from: {folder_url}")
    print(f"Destination: {test_data_dir}")
    
    try:
        gdown.download_folder(url=folder_url, output=test_data_dir, quiet=False, use_cookies=False)
        print(f"\nTest data downloaded successfully to: {test_data_dir}")
        
        # Find and list SVS files
        svs_files = []
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                if file.endswith('.svs'):
                    svs_files.append(os.path.join(root, file))
        
        if not svs_files:
            print("WARNING: No SVS files found in downloaded data")
            return 1
        
        print(f"\nFound {len(svs_files)} SVS file(s):")
        for svs_file in svs_files:
            print(f"  - {svs_file}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to download test data: {e}")
        return 1


if __name__ == "__main__":
    exit_code = download_test_data()
    sys.exit(exit_code)
