import json

import cv2
import mediapipe as mp
import matplotlib

from utils import resize

matplotlib.use('TkAgg')

def pixelate_image(image, pixel_size):
    # Riduci la risoluzione
    small = cv2.resize(image,
                       (image.shape[1] // pixel_size, image.shape[0] // pixel_size),
                       interpolation=cv2.INTER_LINEAR)
    # Ripristina alla risoluzione originale
    pixelated = cv2.resize(small,
                           (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return pixelated

mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    img_path = "C:\\Users\\matte\\PycharmProjects\\TII-Safety\\dataset\\francesco gloves\\guanto non tolto" \
               "\\2024-07-19_052911\\data\\158539.634349291_5.jpeg "

    img = cv2.imread(img_path)
    img = resize(img, 640, 480)
    cv2.imshow('img', img)
    cv2.waitKey()

    pixel_sizes = [2, 4, 8, 16, 32]

    # Applica il filtro a diversi livelli di intensità
    pixelated_images = [pixelate_image(img, size) for size in pixel_sizes]

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for img in pixelated_images:
        cv2.imshow('pixelated', img)
        cv2.waitKey()

    # Get pose landmarks and all detected points
    dic = {}
    for mark, data_point in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
        dic[mark.value] = dict(landmark=mark.name,
                               x=data_point.x,
                               y=data_point.y,
                               z=data_point.z,
                               visibility=data_point.visibility)

    json_object = json.dumps(dic, indent=2)
    print(json_object)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = img.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('ann', annotated_image)
    cv2.waitKey()

    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
