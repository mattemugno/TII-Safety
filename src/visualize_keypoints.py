import os
import json
import cv2
import random

# --- CONFIGURAZIONE --------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.getcwd()))
image_dir = BASE_DIR + '/data/dataset/francesco gloves/2024-07-19_052113/data'
keypoints_json = BASE_DIR + '/data/dataset/francesco gloves/keypoints francesco.json'

output_path = 'output.jpg'

# --- CARICAMENTO KEYPOINTS -------------------------------------------
with open(keypoints_json, 'r') as f:
    kp_dict = json.load(f)

fname = '158057.611983791_5.jpeg'
img_path = os.path.join(image_dir, fname)

# --- LETTURA E DISEGNO -----------------------------------------------
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Non riesco a leggere {img_path}")

h, w = img.shape[:2]

colors = {
    'lh': (0, 255, 0),
    'rh': (0, 0, 255),
    'le': (255, 0, 0),
    're': (255, 255, 0),
    'ls': (0, 255, 255),
    'rs': (255, 0, 255),
}

connections = [
    ('ls', 'le'),
    ('le', 'lh'),
    ('rs', 're'),
    ('re', 'rh'),
    ('ls', 'rs'),
]

# prima disegni i punti (come hai già fatto)
for label, vals in kp_dict[fname]['keypoints'].items():
    x_norm, y_norm, z, vis = vals

    cx = int(x_norm * w)
    cy = int(y_norm * h)

    color = colors.get(label, (200,200,200))
    cv2.circle(img, (cx, cy), radius=5, color=color, thickness=2)
    cv2.putText(img, label, (cx + 5, cy - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2)

for a, b in connections:
    if a in kp_dict[fname]['keypoints'] and b in kp_dict[fname]['keypoints']:
        xa, ya, _, va = kp_dict[fname]['keypoints'][a]
        xb, yb, _, vb = kp_dict[fname]['keypoints'][b]
        if va >= 0.5 and vb >= 0.5:
            pa = (int(xa * w), int(ya * h))
            pb = (int(xb * w), int(yb * h))
            cv2.line(img, pa, pb, color=(255,255,255), thickness=2)

# --- SALVATAGGIO -----------------------------------------------------
cv2.imwrite(output_path, img)
print(f"Immagine annotata salvata in {output_path}")
