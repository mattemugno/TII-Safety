import json
import os

import mediapipe as mp

from utils import get_image_files, get_keypoints

dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'dataset')

subjects = ["francesco", "matteo", "michele", "michela"]
mp_pose = mp.solutions.pose

'''for subject in subjects:
    subject_data_path = os.path.join(dataset_path, subject + " gloves")

    img_files_by_folder = get_image_files(subject_data_path)
    keypoints = {}

    for img_files in img_files_by_folder.values():
        get_keypoints(img_files, mp_pose, keypoints)

    print(f"Keypoints of subject {subject} completed. \n")

    with open(os.path.join(subject_data_path, f'keypoints {subject}.json'), 'w') as f:
        json.dump(keypoints, f, indent=2)'''

for subject in subjects:
    subject_data_path = os.path.join(dataset_path, subject + " tornio safe")

    img_files_by_folder = get_image_files(subject_data_path)
    keypoints = {}

    for img_files in img_files_by_folder.values():
        get_keypoints(img_files, mp_pose, keypoints)

    print(f"Keypoints of subject {subject} completed. \n")

    with open(os.path.join(subject_data_path, f'keypoints {subject}.json'), 'w') as f:
        json.dump(keypoints, f, indent=2)