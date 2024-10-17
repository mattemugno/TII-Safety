import json
import os
from os.path import abspath, dirname

import mediapipe as mp

from utils import get_image_files, read_json, get_bounding_boxes

dataset_path = "../dataset"

subjects = ["francesco", "matteo", "michele", "michela"]
mp_pose = mp.solutions.pose

for subject in subjects:
    subject_gloves_path = os.path.join(dataset_path, '../dataset/gloves_task', 'gloves_' + f'{subject}')
    data_subject_path = os.path.join(dataset_path, subject + " tornio disp corr")

    # if not os.path.exists(os.path.join(dirname(abspath(__file__)), subject_gloves_path)):
    #     os.makedirs(os.path.join(dirname(abspath(__file__)), subject_gloves_path))

    if not os.path.exists(subject_gloves_path):
        os.makedirs(subject_gloves_path)

    img_files = get_image_files(data_subject_path)

    keypoints = os.path.join(data_subject_path, f'keypoints {subject}.json')
    keypoints = read_json(keypoints)

    bboxes = {}

    for image_path in img_files:
        get_bounding_boxes(image_path, keypoints, subject_gloves_path, save=True, bboxes=bboxes)

    print(f"BBoxes of subject {subject} completed. \n")

    with open(os.path.join(data_subject_path, 'bboxes.json'), 'w') as f:
        json.dump(bboxes, f)
