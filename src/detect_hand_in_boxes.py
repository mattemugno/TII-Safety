import os
import json
import cv2
import mediapipe as mp

from utils import detect_hand_in_image

# --- CONFIGURAZIONE DI BASE -----------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
SUBJECTS = ["francesco", "matteo", "michele", "michela"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

PATCH_RADIUS = 50

for subject in SUBJECTS:
    subj_folder = os.path.join(DATASET_PATH, f'{subject} gloves')

    # carica i keypoints già calcolati
    kp_path = os.path.join(subj_folder, f'keypoints {subject}.json')
    with open(kp_path, 'r') as f:
        kp_dict = json.load(f)

    annotations = {}

    for session in os.listdir(subj_folder):
        if os.path.isdir(os.path.join(subj_folder, session)):
            session_path = os.path.join(subj_folder, session)
            data_dir = os.path.join(session_path, 'data')

            for fname in os.listdir(data_dir):
                full_img_path = os.path.join(data_dir, fname)
                img = cv2.imread(full_img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                entry = [0, 0]  # [left, right]

                if fname in kp_dict:
                    kp_info = kp_dict[fname]['keypoints']

                    # iterate on left hand (lh) and right hand (rh)
                    for idx, hand_label in enumerate(('lh', 'rh')):
                        if hand_label not in kp_info:
                            continue
                        x_norm, y_norm, _, vis = kp_info[hand_label]
                        if vis < 0.5:
                            continue

                        cx = int(x_norm * w)
                        cy = int(y_norm * h)
                        x1 = max(cx - PATCH_RADIUS, 0)
                        x2 = min(cx + PATCH_RADIUS, w)
                        y1 = max(cy - PATCH_RADIUS, 0)
                        y2 = min(cy + PATCH_RADIUS, h)
                        patch = img[y1:y2, x1:x2]

                        entry[idx] = 0 if detect_hand_in_image(hands, patch) else 1

                annotations[fname] = entry

    out_path = os.path.join(subj_folder, f'annotations_{subject}.json')
    with open(out_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"[✔] Annotations for {subject} written to {out_path}")
