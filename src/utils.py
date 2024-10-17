import math
import os
import json
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions


def get_image_files(img_path):
    image_files = []
    for root, dirs, files in os.walk(img_path):
        for dir in dirs:
            for file in os.listdir(os.path.join(root, dir)):
                if file.endswith('.jpeg'):
                    image_files.append(os.path.join(root, dir, file))
    return image_files


def resize(image, DESIRED_WIDTH, DESIRED_HEIGHT):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    return img


def read_depth_map_from_json(file_path):
    """Reads the 'depth map' field from a JSON file.

  Args:
    file_path: The path to the JSON file.

  Returns:
    The 'depth map' field as a list, or None if the field is not found.
  """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'depthMap' in data:
                return np.array(data['depthMap'])
            else:
                return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def read_json(file_path):
    """Reads a JSON file.

  Args:
    file_path: The path to the JSON file.

  Returns:
    The 'depth map' field as a list, or None if the field is not found.
  """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def get_keypoints(image_files, mp_pose, keypoints):
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        for im_name in image_files:
            image = cv2.imread(im_name)

            name = im_name.split('\\')[-1]

            image = resize(image, 256, 192)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # image_height, _ = image.shape

            if not results.pose_landmarks:
                continue

            depth_map = read_depth_map_from_json(im_name.replace('.jpeg', '.json'))
            depth_img = np.stack((depth_map,) * 3, axis=-1)

            keypoints[name] = {}

            # Check keypoint visibility > 0.5

            keypoints[name]['lh'] = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                                     depth_img[int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.LEFT_WRIST].y),
                                               int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.LEFT_WRIST].x)][0],
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility)

            keypoints[name]['rh'] = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                                     depth_img[int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.RIGHT_WRIST].y),
                                               int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.RIGHT_WRIST].x)][0])

            keypoints[name]['le'] = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                                     depth_img[int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.LEFT_ELBOW].y),
                                               int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.LEFT_ELBOW].x)][0])

            keypoints[name]['re'] = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                                     depth_img[int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.RIGHT_ELBOW].y),
                                               int(results.pose_landmarks.landmark[
                                                       mp_pose.PoseLandmark.RIGHT_ELBOW].x)][0])

            keypoints[name]['ls'] = (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                depth_img[int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
                          int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)][0])

            keypoints[name]['rs'] = (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                depth_img[int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
                          int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)][0])

            print(f'Image {im_name} processed.')


def detect_hand_in_image(hands, image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        return True
    else:
        return False


def get_bounding_boxes(image_path, keypoints_data, save_path, save=True, bboxes=None):
    """Draws bounding boxes around hands based on keypoints and saves the boxes as new images.

  Args:
    :param image_path: Path to the image file.
    :param bboxes: Dictionary containing bboxes data.
    :param keypoints_data: Dictionary keypoints data.
    :param save: Bool for saving or not images.
    :param save_path: Save path.
  """

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return

        image_name = os.path.basename(image_path)
        if bboxes is not None:
            bboxes[image_name] = {}

        if image_name in keypoints_data:
            keypoints = keypoints_data[image_name]

            h, w = image.shape[:2]

            if 'lh' in keypoints and keypoints['lh']:
                x, y = keypoints['lh'][0], keypoints['lh'][1]
                x = int(x * w)
                y = int(y * h)

                box = image[max(0, y - 150):min(h, y + 150), max(0, x - 150):min(w, x + 150)]
                if bboxes is not None:
                    bboxes[image_name]['lh'] = [max(0, x - 150), max(0, y - 150), min(w, x + 150), min(h, y + 150)]

                if save:
                    box_filename = f"{image_name[:-5]}_l_box.jpeg"
                    box = resize(box, 192, 192)
                    cv2.imwrite(os.path.join(save_path, box_filename), box)

            if 'rh' in keypoints and keypoints['rh']:
                x, y = keypoints['rh'][0], keypoints['rh'][1]
                x = int(x * w)
                y = int(y * h)

                box = image[max(0, y - 150):min(h, y + 150), max(0, x - 150):min(w, x + 150)]
                if bboxes is not None:
                    bboxes[image_name]['rh'] = [max(0, x - 150), max(0, y - 150), min(w, x + 150), min(h, y + 150)]

                if save:
                    box_filename = f"{image_name[:-5]}_r_box.jpeg"
                    box = resize(box, 192, 192)
                    cv2.imwrite(os.path.join(save_path, box_filename), box)

                print(f'Image {image_path} processed.')

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
