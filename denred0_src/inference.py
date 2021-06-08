import os
import cv2
import sys
import time
import torch

import numpy as np

import albumentations as A

from pathlib import Path

from hand_detection import get_hands_rects
from face_detection import get_face_rect

from model import Model
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), ToTensorV2()])


def get_model(checkpoint):
    model = Model.load_from_checkpoint(checkpoint_path=best_checkpoint)
    model = model.to("cuda")
    model.eval()
    model.freeze()

    return model


def inference(video_src_dir, video_dst_dir, video_name, checkpoint, image_size_model, frame_rate=18):
    cap = cv2.VideoCapture(str(Path(video_src_dir).joinpath(video_name)))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('fps', fps)

    time_total = 0

    print('frames_count', length)

    if not cap.isOpened():
        print("Error opening the video file. Please double check your "
              "file path for typos. Or move the movie file to the same location as this script/notebook")
        sys.exit()

    img_array = []
    size_shape = ()

    model = get_model(checkpoint=checkpoint)

    while cap.isOpened():
        # Read the video file.
        ret, image = cap.read()

        # If we got frames, show them.
        if ret:

            start = time.time()
            # time.sleep(1 / fps)

            # decrease resolution of video
            scale_percent = 50  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            crops_for_prediction = {}
            # get hands rects if hands found
            hands_rects = get_hands_rects(image, min_detection_confidence=0.4)
            for i, rect in enumerate(hands_rects):
                crops_for_prediction['hand_' + str(i)] = rect
                # crops_for_prediction.append(rect)

            if crops_for_prediction:
                print()
            face_rect = get_face_rect(image, min_detection_confidence=0.4)
            if face_rect:
                crops_for_prediction['face'] = face_rect
                # image = cv2.rectangle(image, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (255, 255, 255), 6)

                # crops_for_prediction.append(face_rect)

            if face_rect:
                print()
            print('face_rect', face_rect)

            for key, rect in crops_for_prediction.items():

                # for rect in crops_for_prediction:

                x = rect[0]
                y = rect[1]
                x2 = rect[2]
                y2 = rect[3]

                hand_img = image[y:y2, x:x2]

                hand_img_resized = cv2.resize(hand_img, (image_size_model, image_size_model),
                                              interpolation=cv2.INTER_AREA)

                hand_img_resized = transforms(image=hand_img_resized)
                hand_img_resized = torch.unsqueeze(hand_img_resized.get('image'), 0)

                y_hat = model(hand_img_resized.to('cuda'))

                y_hat = y_hat.cpu().detach().numpy()[0]

                y_hat_hand = y_hat[0:2]
                y_hat_face = y_hat[2:4]

                if key.startswith('hand'):
                    if np.argmax(y_hat_hand) == 1:
                        image = cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 6)

                if key.startswith('face'):
                    if np.argmax(y_hat_face) == 1:
                        image = cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 255), 6)

            # show frame
            cv2.imshow('image', image)

            (H, W) = image.shape[:2]
            size_shape = (W, H)

            # for video creation
            # img_array.append(image)

            end = time.time()

            print(str(end - start))
            time_total += end - start

            if cv2.waitKey(int(1000./float(fps))) & 0xFF == ord('q'):
                break

        else:
            break

    print((time_total) / length)
    # write video
    print("Saving video...")
    out = cv2.VideoWriter(str(Path(video_dst_dir).joinpath(video_name + '_result.avi')),
                          cv2.VideoWriter_fourcc(*'DIVX'), frame_rate,
                          size_shape)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_src_dir = os.path.join('denred0_data', 'inference', 'source')
    video_dst_dir = os.path.join('denred0_data', 'inference', 'result')

    video_name = 'tst_bio.avi'  # 'tst_bio.avi'

    best_checkpoint = os.path.join('denred0_model', 'best_checkpoint',
                                   'senet154_hands_epoch=6_val_loss=0.015_val_acc=0.996_val_f1_epoch=0.996.ckpt')
    image_size_model = 224

    frame_rate = 36  # frame rate in resulting video

    inference(video_src_dir=video_src_dir,
              video_dst_dir=video_dst_dir,
              video_name=video_name,
              checkpoint=best_checkpoint,
              image_size_model=image_size_model,
              frame_rate=frame_rate)
