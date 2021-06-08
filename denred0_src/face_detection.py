import cv2
import mediapipe as mp
import os
from tqdm import tqdm

from mediapipe2.python.solutions import face_mesh as mediapipe_face_mesh

from os import walk


def get_face_rect(image, min_detection_confidence):
    face_rect = []

    with mediapipe_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = image.shape

        # if results.multi_face_landmarks:
        decrease_box = 0

        rect = results.face_rect
        if rect:
            x = abs(int((rect.x_center * image_width - rect.width * image_width / 2) * (1 + decrease_box)))
            y = abs(int((rect.y_center * image_height - rect.height * image_height / 2) * (1 + decrease_box)))

            x2 = abs(int((rect.x_center * image_width + rect.width * image_width / 2) * (1 - decrease_box)))
            y2 = abs(int((rect.y_center * image_height + rect.height * image_height / 2) * (1 - decrease_box)))

            face_rect = [x, y, x2, y2]

    return face_rect


def example():
    # mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mediapipe_face_mesh

    data_dir = 'denred0_data'
    tag = 'tst_bio'

    # For static images:
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        _, _, file_list = next(walk(os.path.join(data_dir, 'face_source')))

        for idx, file in tqdm(enumerate(sorted(file_list))):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(os.path.join(data_dir, 'face_source', file)), 1)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            # print('Handedness:', results.multi_handedness)
            if not results.multi_face_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            rect = results.face_rect
            # for i, rect in enumerate(results.face_rect):
            color = (255, 0, 255)
            # if results.multi_handedness[i].classification[0].label == 'Right':
            #     color = (255, 0, 255)
            # else:
            #     color = (0, 255, 255)

            thickness = 6

            decrease_box = 0

            x = abs(int((rect.x_center * image_width - rect.width * image_width / 2) * (1 + decrease_box)))
            y = abs(int((rect.y_center * image_height - rect.height * image_height / 2) * (1 + decrease_box)))

            x2 = abs(int((rect.x_center * image_width + rect.width * image_width / 2) * (1 - decrease_box)))
            y2 = abs(int((rect.y_center * image_height + rect.height * image_height / 2) * (1 - decrease_box)))

            hand_img = image[y:y2, x:x2]

            cv2.imwrite(data_dir + '/face_crop/' + tag + '_' + str(idx) + '.jpg', cv2.flip(hand_img, 1))

            annotated_image = cv2.rectangle(annotated_image, (x, y), (x2, y2), color, thickness)

            cv2.imwrite(data_dir + '/face_detection/' + tag + '_' + str(idx) + '.jpg', cv2.flip(annotated_image, 1))

            # annotated_image = cv2.rectangle(annotated_image, (x, y), (x2, y2), color, thickness)

            # for hand_landmarks in results.multi_hand_landmarks:
            #     print('hand_landmarks:', hand_landmarks)
            #     print(
            #         f'Index finger tip coordinates: (',
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            #     )
            #     mp_drawing.draw_landmarks(
            #         annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


if __name__ == '__main__':
    example()
