import pandas as pd
from tqdm import tqdm, trange

export_json_path = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/Export_catalog_query_3_10_2025.ndjson'
# export_json_path = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/data/labelled_images/Export_catalog_query_3_10_2025_1_img.ndjson'

export_df = pd.read_json(export_json_path, lines=True)
# export_df = pd.DataFrame(export_df['data_row'].values.tolist())

filename_list = []
polygon_list = []
for row_ind in trange(len(export_df)):
    # this_row = export_df.iloc[0]
    this_row = export_df.iloc[row_ind]
    file_name = this_row['data_row']['external_id']
    projects = this_row['projects']
    project_keys = list(projects.keys())

    # this_key = project_keys[0]
    for this_key in project_keys:
        labels = export_df.iloc[0]['projects'][this_key]['labels']

        for this_label in labels:
            objects = this_label['annotations']['objects']
            for this_object in objects:
                polygon = this_object['polygon']
                filename_list.append(file_name)
                polygon_list.append(polygon)

polygon_df = pd.DataFrame({'filename':filename_list, 'polygon':polygon_list})
