import os
import csv

import pandas as pd

from utils import read_json


def frame_to_row(sequence_id, frame, sequence):
    timestamp = frame.split('_')[0]
    keypoints = sequence[frame]['keypoints']
    gloves = sequence[frame]['gloves_flag']
    occlusion = sequence[frame]['occlusion_flags']

    return [
        sequence_id,
        timestamp,
        keypoints['lh'][0], keypoints['lh'][1], keypoints['lh'][2],
        keypoints['rh'][0], keypoints['rh'][1], keypoints['rh'][2],
        keypoints['le'][0], keypoints['le'][1], keypoints['le'][2],
        keypoints['re'][0], keypoints['re'][1], keypoints['re'][2],
        keypoints['ls'][0], keypoints['ls'][1], keypoints['ls'][2],
        keypoints['rs'][0], keypoints['rs'][1], keypoints['rs'][2],
        gloves['g_lh'], gloves['g_rh'],
        occlusion['o_lh'], occlusion['o_rh']
    ]


def convert_json_to_csv(output_file_path, subjects):
    with open(output_file_path, 'w', newline='') as csvfile:
        output_writer = csv.writer(csvfile)

        # Write header
        output_writer.writerow([
            "Sequence ID", "Timestamp",
            "LH_X", "LH_Y", "LH_Z",
            "RH_X", "RH_Y", "RH_Z",
            "LE_X", "LE_Y", "LE_Z",
            "RE_X", "RE_Y", "RE_Z",
            "LS_X", "LS_Y", "LS_Z",
            "RS_X", "RS_Y", "RS_Z",
            "Gloves_LH", "Gloves_RH",
            "Occlusion_LH", "Occlusion_RH"
        ])

        for subject in subjects:
            subject_data_path = os.path.join(dataset_path, subject + " gloves")
            data_path = os.path.join(subject_data_path, f'keypoints {subject}.json')
            data = read_json(data_path)

            for seq_id, sequence_data in data.items():
                for frame in sequence_data:
                    row = frame_to_row(seq_id.split('\\')[-1], frame, sequence_data)
                    output_writer.writerow(row)


dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
subjects = ["francesco", "matteo", "michele", "michela"]

output_file_path = os.path.join(dataset_path, 'data.csv')
convert_json_to_csv(output_file_path, subjects)
print(f"Conversion complete! All data has been saved at: {output_file_path}")

df = pd.read_csv(output_file_path, index_col=None)
print("Shape: " + str(df.shape))
print("Head: " + str(df.head()))
