import os
import json
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp

# --- BASE CONFIG -----------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
SUBJECTS = ["francesco", "matteo", "michele", "michela"]
PATCH_RADIUS = 100

# --- CLIP SETUP ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
prompts = ["a glove", "a bare hand"]

# --- MEDIAPIPE HANDS SETUP -----------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def detect_glove_in_patch(patch: "ndarray") -> bool:
    """
    Classify whether a cropped hand image is wearing a glove using CLIP zero-shot.
    Returns True if glove, False otherwise.
    """
    if patch is None or patch.size == 0:
        return False

    img_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    inputs = clip_processor(text=prompts, images=img_rgb, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

    return bool(probs[0] > probs[1])


def crop_and_label(img, kp, w, h):
    """
    Crop around a keypoint and label glove vs no_glove.
    Returns the patch and boolean glove flag.
    """
    x_norm, y_norm, _, vis = kp
    if vis < 0.5:
        return None, False
    cx = int(x_norm * w)
    cy = int(y_norm * h)
    x1 = max(cx - PATCH_RADIUS, 0)
    x2 = min(cx + PATCH_RADIUS, w)
    y1 = max(cy - PATCH_RADIUS, 0)
    y2 = min(cy + PATCH_RADIUS, h)
    patch = img[y1:y2, x1:x2]
    has_glove = detect_glove_in_patch(patch)
    return patch, has_glove


for subject in SUBJECTS:
    subj_folder = os.path.join(DATASET_PATH, f'{subject} gloves')
    safe_dir = os.path.join(subj_folder, 'safe')
    unsafe_dir = os.path.join(subj_folder, 'unsafe')
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)

    kp_path = os.path.join(subj_folder, f'keypoints {subject}.json')
    with open(kp_path, 'r') as f:
        kp_dict = json.load(f)

    annotations = {}

    for session in os.listdir(subj_folder):
        session_path = os.path.join(subj_folder, session)
        data_dir = os.path.join(session_path, 'data')
        if not os.path.isdir(data_dir):
            continue

        for fname in os.listdir(data_dir):
            full_img_path = os.path.join(data_dir, fname)
            img = cv2.imread(full_img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            entry = [0, 0]  # [left_glove, right_glove]

            if fname in kp_dict:
                kp_info = kp_dict[fname]['keypoints']

                for idx, hand_label in enumerate(('lh', 'rh')):
                    if hand_label not in kp_info:
                        continue
                    patch, has_glove = crop_and_label(img, kp_info[hand_label], w, h)
                    entry[idx] = 1 if has_glove else 0

                    # Save patch in corresponding folder
                    out_folder = safe_dir if has_glove else unsafe_dir
                    out_name = f"{subject}_{session}_{fname.replace('.', '_')}_{hand_label}.png"
                    out_path = os.path.join(out_folder, out_name)
                    if patch is not None:
                        cv2.imwrite(out_path, patch)

            annotations[fname] = entry

    out_ann = os.path.join(subj_folder, f'annotations_{subject}.json')
    with open(out_ann, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"[✔] Annotations and patches saved for {subject}")
