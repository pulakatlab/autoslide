import labelbox as lb
import os
from autoslide.pipeline import utils

api_key = os.getenv("LABELBOX_API_KEY")
client = lb.Client(api_key)

dataset = client.get_dataset("cmc9jjmac01ly0742nzmdhqmk")

data_dir = "/media/bigdata/projects/auto_slide/data/suggested_regions/heart_sections"
file_list = os.listdir(data_dir)

file_paths = [os.path.join(data_dir, file_name) for file_name in file_list]
hash_list = [utils.extract_hash_from_filename(x) for x in file_list]

assets = []
for file_path, hash_value in zip(file_paths, hash_list):
    asset = {
        "row_data": file_path,
        "global_key": hash_value,
        "media_type": "IMAGE",
    }
    assets.append(asset)

task = dataset.create_data_rows(assets)
task.wait_till_done()
print(task.errors)
