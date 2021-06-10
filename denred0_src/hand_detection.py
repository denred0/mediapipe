import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from pathlib import Path

from os import walk

from mediapipe2.python.solutions import hands as mediapipe_hands

from os import walk


def get_hands_rects(image, min_detection_confidence):
    hands_rects = []

    with mediapipe_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for i, rect in enumerate(results.hand_rects):
                decrease_box = 0

                x = abs(int((rect.x_center * image_width - rect.width * image_width / 2) * (1 + decrease_box)))
                y = abs(int((rect.y_center * image_height - rect.height * image_height / 2) * (1 + decrease_box)))

                x2 = abs(int((rect.x_center * image_width + rect.width * image_width / 2) * (1 - decrease_box)))
                y2 = abs(int((rect.y_center * image_height + rect.height * image_height / 2) * (1 - decrease_box)))

                hand_rest = [x, y, x2, y2]
                hands_rects.append(hand_rest)

    return hands_rects


def example():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mediapipe_hands

    data_dir = 'denred0_data'
    tag = 'tst_bio'

    # For static images:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.4) as hands:
        _, _, file_list = next(walk(os.path.join(data_dir, 'hands_source')))

        for idx, file in tqdm(enumerate(sorted(file_list))):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(os.path.join(data_dir, 'hands_source', file)), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            # print('Handedness:', results.multi_handedness)
            if not results.hand_rects:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            for i, rect in enumerate(results.hand_rects):
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

                cv2.imwrite(data_dir + '/hands_crop/' + tag + '_' + str(idx) + '_' + str(i) + '.jpg',
                            cv2.flip(hand_img, 1))

                annotated_image = cv2.rectangle(annotated_image, (x, y), (x2, y2), color, thickness)

            cv2.imwrite(data_dir + '/hand_detection/' + tag + '_' + str(idx) + '.jpg', cv2.flip(annotated_image, 1))

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


def example_comp():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mediapipe_hands

    data_dir = 'data_comp/hands_comp'
    tag = 'tst_bio'

    # For static images:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=6,
            min_detection_confidence=0.6) as hands:

        for subdir, dirs, files in os.walk(data_dir):
            for k, folder in tqdm(enumerate(dirs), total=len(dirs)):
                Path('data_comp/hands_rect/' + str(folder)).mkdir(parents=True, exist_ok=True)
                Path('data_comp/hands_crop/' + str(folder)).mkdir(parents=True, exist_ok=True)

                p = data_dir + '/' + folder + '/'
                _, _, images_list = next(walk(p))

                for file in images_list:

                    filename, file_extension = os.path.splitext(file)

                    # Read an image, flip it around y-axis for correct handedness output (see
                    # above).
                    image = cv2.flip(cv2.imread(os.path.join(data_dir, folder, file)), 1)
                    # Convert the BGR image to RGB before processing.
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    # Print handedness and draw hand landmarks on the image.
                    # print('Handedness:', results.multi_handedness)
                    if not results.hand_rects:
                        continue
                    image_height, image_width, _ = image.shape
                    annotated_image = image.copy()

                    for i, rect in enumerate(results.hand_rects):
                        color = (255, 0, 255)
                        # if results.multi_handedness[i].classification[0].label == 'Right':
                        #     color = (255, 0, 255)
                        # else:
                        #     color = (0, 255, 255)

                        thickness = 6

                        decrease_box = 0

                        x = abs(int((rect.x_center * image_width - rect.width * image_width / 2) * (1 + decrease_box)))
                        y = abs(
                            int((rect.y_center * image_height - rect.height * image_height / 2) * (1 + decrease_box)))

                        x2 = abs(int((rect.x_center * image_width + rect.width * image_width / 2) * (1 - decrease_box)))
                        y2 = abs(
                            int((rect.y_center * image_height + rect.height * image_height / 2) * (1 - decrease_box)))

                        hand_img = image[y:y2, x:x2]

                        cv2.imwrite('data_comp/hands_crop/' + folder + '/' + filename + '_' + str(i) + file_extension,
                                    cv2.flip(hand_img, 1))

                        annotated_image = cv2.rectangle(annotated_image, (x, y), (x2, y2), color, thickness)

                    cv2.imwrite('data_comp/hands_rect/' + folder + '/' + file, cv2.flip(annotated_image, 1))


if __name__ == '__main__':
    example_comp()
