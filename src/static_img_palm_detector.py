import cv2
import argparse
import os
from src.hand_tracker import HandTracker

# USAGE: python run.py --3d [true/false] --output_path [folder]
ap = argparse.ArgumentParser()
ap.add_argument("--3d", required=True, help="Check for type of detection")
ap.add_argument("--output_path", required=True, help="Path to save the processed images")
args = vars(ap.parse_args())

PALM_MODEL_PATH = "hand_tracking/models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "hand_tracking/models/hand_landmark.tflite"
ANCHORS_PATH = "hand_tracking/models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

root_folder = "dataset/boxes michela-20241016T073053Z-001"
image_files = []
for root, dirs, files in os.walk(root_folder):
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith('.jpeg'):
                image_files.append(os.path.join(root, dir, file))

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

hand_3d = args["3d"]
output_path = args["output_path"]

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize hand tracker
detector = HandTracker(
    hand_3d,
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)

for idx, image_path in enumerate(image_files):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error loading image {image_path}")
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)

    if points is not None:
        if hand_3d == "True":
            for point in points:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            for connection in connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        else:
            cv2.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), CONNECTION_COLOR,
                     THICKNESS)
            cv2.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), CONNECTION_COLOR,
                     THICKNESS)
            cv2.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), CONNECTION_COLOR,
                     THICKNESS)
            cv2.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), CONNECTION_COLOR,
                     THICKNESS)

            # Save the processed image to the output folder
            output_file_path = os.path.join(output_path, f"processed_image_{idx}.jpg")
            cv2.imwrite(output_file_path, frame)
            print(f"Saved processed image to {output_file_path}")
