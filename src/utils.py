import json
import math
import os

import cv2
import numpy as np

VISIBILITY_THRESHOLD = 0.5


def get_image_files(img_path):
    image_files = {}
    for root, dirs, files in os.walk(img_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    data_path = os.path.join(subdir_path, 'data')
                    if os.path.isdir(data_path):
                        image_files[subdir_path] = []
                        for file in os.listdir(data_path):
                            if file.endswith('.jpeg'):
                                image_files[subdir_path].append(os.path.join(data_path, file))
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


def round_tuple(values):
    return tuple(round(v, 4) for v in values)


def get_keypoints(image_files, mp_pose, keypoints, folder):
    folder = os.path.relpath(folder, start=os.path.join(os.getcwd(), 'dataset'))

    if folder not in keypoints:
        keypoints[folder] = {}

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

            keypoints[folder][name]['keypoints'] = {}

            keypoints[folder][name]['keypoints']['lh'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_WRIST].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_WRIST].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.LEFT_WRIST].visibility))

            keypoints[folder][name]['keypoints']['rh'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_WRIST].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_WRIST].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.RIGHT_WRIST].visibility))

            keypoints[folder][name]['keypoints']['le'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_ELBOW].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_ELBOW].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.LEFT_ELBOW].visibility))

            keypoints[folder][name]['keypoints']['re'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_ELBOW].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_ELBOW].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.RIGHT_ELBOW].visibility))

            keypoints[folder][name]['keypoints']['ls'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_SHOULDER].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.LEFT_SHOULDER].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.LEFT_SHOULDER].visibility))

            keypoints[folder][name]['keypoints']['rs'] = round_tuple(
                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                 depth_img[int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
                           int(results.pose_landmarks.landmark[
                                   mp_pose.PoseLandmark.RIGHT_SHOULDER].x)][0],
                 results.pose_landmarks.landmark[
                     mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility))

            print(f'Image {im_name} processed.')


def detect_hand_in_image(hands, image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("The image does not exist.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            return True
        else:
            return False

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_bounding_boxes(folder, image_name, keypoints_data, save_path, save=True, bboxes=None):
    """Draws bounding boxes around hands based on keypoints and saves the boxes as new images.

    :param folder: The folder name where the image is located within the dataset.
    :param image_name: The name of the image file to be processed.
    :param keypoints_data: Dictionary containing keypoints data with the structure
                            keypoints_data[folder][image_name]['lh'] or ['rh'], where 'lh' refers to left hand
                            and 'rh' to right hand. Each contains keypoints and a visibility score.
    :param save_path: The path where the cropped images with bounding boxes will be saved.
    :param save: If True, saves the cropped images with bounding boxes. Default is True.
    :param bboxes: Dictionary to store bounding box coordinates for each image. If None, no bounding
                            box information will be stored. Structure should be bboxes[folder][image_name]['lh']
                            or ['rh'].
    :returns: None: The function processes the image, draws bounding boxes if keypoints are visible, and optionally
                    saves the cropped images and bounding box data.
  """

    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    image_path = os.path.join(dataset_path, folder, 'data', image_name)

    try:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            return

        if bboxes is not None and folder not in bboxes:
            bboxes[folder] = {}

        bboxes[folder][image_name] = {}

        keypoints = keypoints_data[folder][image_name]['keypoints']
        h, w = image.shape[:2]

        if 'lh' in keypoints and keypoints['lh'][3] > VISIBILITY_THRESHOLD:
            x, y = keypoints['lh'][0], keypoints['lh'][1]
            x = int(x * w)
            y = int(y * h)
            box = image[max(0, y - 150):min(h, y + 150), max(0, x - 150):min(w, x + 150)]
            if bboxes is not None:
                bboxes[folder][image_name]['lh'] = [max(0, x - 150), max(0, y - 150), min(w, x + 150),
                                                    min(h, y + 150)]
            if save:
                box_filename = f"{image_name[:-5]}_l_box.jpeg"
                box = resize(box, 192, 192)
                cv2.imwrite(os.path.join(save_path, box_filename), box)

        if 'rh' in keypoints and keypoints['rh'][3] > VISIBILITY_THRESHOLD:
            x, y = keypoints['rh'][0], keypoints['rh'][1]
            x = int(x * w)
            y = int(y * h)
            box = image[max(0, y - 150):min(h, y + 150), max(0, x - 150):min(w, x + 150)]
            if bboxes is not None:
                bboxes[folder][image_name]['rh'] = [max(0, x - 150), max(0, y - 150), min(w, x + 150),
                                                    min(h, y + 150)]
            if save:
                box_filename = f"{image_name[:-5]}_r_box.jpeg"
                box = resize(box, 192, 192)
                cv2.imwrite(os.path.join(save_path, box_filename), box)

        print(f'Image {image_path} processed.')

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def set_metadata(folder, img_name, keypoints, subject, hands):
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')

    try:
        metadata = keypoints[folder][img_name]

        # metadata['o_lh'] = 0 left hand occluded
        # metadata['o_lh'] = 1 left hand not occluded
        metadata['occlusion_flags'] = {}
        metadata['occlusion_flags']['o_lh'] = 0 if metadata['lh'][3] < VISIBILITY_THRESHOLD else 1
        metadata['occlusion_flags']['o_rh'] = 0 if metadata['rh'][3] < VISIBILITY_THRESHOLD else 1

        # if not occluded then check if hand wears glove or not
        # metadata['g_lh'] = 1 left bare hand
        # metadata['g_lh'] = 0 left gloved hand
        # metadata['g_lh'] = -1 left hand occluded
        metadata['gloves_flag'] = {}
        if metadata['occlusion_flags']['o_lh']:
            box_name = os.path.join(dataset_path, 'gloves_task', f'gloves_{subject}', f"{img_name[:-5]}_l_box.jpeg")
            metadata['gloves_flag']['g_lh'] = 1 if detect_hand_in_image(hands, box_name) else 0
            print(f'Image {box_name} processed.')
        else:
            metadata['gloves_flag']['g_lh'] = -1

        if metadata['occlusion_flags']['o_rh']:
            box_name = os.path.join(dataset_path, 'gloves_task', f'gloves_{subject}', f"{img_name[:-5]}_r_box.jpeg")
            metadata['gloves_flag']['g_rh'] = 1 if detect_hand_in_image(hands, box_name) else 0
            print(f'Image {box_name} processed.')
        else:
            metadata['gloves_flag']['g_rh'] = -1

    except Exception as e:
        print(f"Error processing image {img_name}: {e}")
