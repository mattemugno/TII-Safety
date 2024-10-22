import json
import os

import mediapipe as mp

from utils import read_json, get_bounding_boxes

dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

subjects = ["francesco", "matteo", "michele", "michela"]
mp_pose = mp.solutions.pose

for subject in subjects:
    subject_gloves_path = os.path.join(dataset_path, 'gloves_task', 'gloves_' + f'{subject}')
    data_subject_path = os.path.join(dataset_path, subject + " gloves")

    if not os.path.exists(subject_gloves_path):
        os.makedirs(subject_gloves_path)

    keypoints = os.path.join(data_subject_path, f'keypoints {subject}.json')
    keypoints = read_json(keypoints)

    bboxes = {}

    for folder, images_path in keypoints.items():
        for img in images_path:
            get_bounding_boxes(folder, img, keypoints, subject_gloves_path, save=True, bboxes=bboxes)

    print(f"BBoxes of subject {subject} completed. \n")

    with open(os.path.join(data_subject_path, 'bboxes.json'), 'w') as f:
        json.dump(bboxes, f, indent=2)
