import json
import os

import mediapipe as mp

from utils import read_json, set_metadata

dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

subjects = ["matteo", "michele", "michela"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

for subject in subjects:
    subject_data_path = os.path.join(dataset_path, subject + " gloves")

    keypoints = os.path.join(subject_data_path, f'keypoints {subject}.json')
    keypoints = read_json(keypoints)

    for seq_id, images in keypoints.items():
        for img_name in images:
            set_metadata(seq_id, img_name, keypoints, subject, hands)

    print(f"Metadata of subject {subject} completed. \n")

    with open(os.path.join(subject_data_path, f'keypoints {subject}.json'), 'w') as f:
        json.dump(keypoints, f, indent=2)
