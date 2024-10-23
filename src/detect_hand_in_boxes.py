import os
import shutil

import mediapipe as mp

from utils import detect_hand_in_image

dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

subjects = ["francesco", "matteo", "michele", "michela"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

for subject in subjects:

    subject_gloves_path = os.path.join(dataset_path, 'gloves_task', 'gloves_' + f'{subject}')

    positive_folder = os.path.join(subject_gloves_path, "positive")
    negative_folder = os.path.join(subject_gloves_path, "negative")

    if not os.path.exists(positive_folder) and not os.path.exists(negative_folder):
        os.makedirs(positive_folder)
        os.makedirs(negative_folder)

    for filename in os.listdir(subject_gloves_path):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(subject_gloves_path, filename)
            if detect_hand_in_image(hands, image_path):
                shutil.move(image_path, positive_folder)
            else:
                shutil.move(image_path, negative_folder)

            print(f'Image {image_path} labelled.')
