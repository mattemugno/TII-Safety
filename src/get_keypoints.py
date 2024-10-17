import json
import os

import mediapipe as mp

from utils import get_image_files, get_keypoints

# ap = argparse.ArgumentParser()
# ap.add_argument("--output_path", required=True, help="Path to save the processed images")
# args = vars(ap.parse_args())

dataset_path = "../dataset"

subjects = ["francesco", "matteo", "michele", "michela"]
mp_pose = mp.solutions.pose

for subject in subjects:
    subject_data_path = os.path.join(dataset_path, subject + " tornio disp corr")

    img_files = get_image_files(subject_data_path)

    keypoints = {}

    get_keypoints(img_files, mp_pose, keypoints)

    print(f"Keypoints of subject {subject} completed. \n")

    with open(os.path.join(subject_data_path, f'keypoints {subject}.json'), 'w') as f:
        json.dump(keypoints, f)
