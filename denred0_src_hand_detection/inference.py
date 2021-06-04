import os
import cv2
import sys
import time
import torch

import albumentations as A

from pathlib import Path

from hand_detection import get_hands_rects
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


def inference(video_src_dir, video_dst_dir, video_name, checkpoint, image_size_model):
    cap = cv2.VideoCapture(str(Path(video_src_dir).joinpath(video_name)))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
            # time.sleep(1 / fps)

            # get hands rects if hands found
            hands_rects = get_hands_rects(image, min_detection_confidence=0.4)
            print()

            for rect in hands_rects:
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
                y_hat = torch.argmax(y_hat, dim=1)

                y_hat = y_hat.cpu().detach().numpy()[0]

                print('y_hat', y_hat)

                if y_hat == 1:
                    image = cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 6)

            # decrease resolution of video
            scale_percent = 50  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # show frame
            cv2.imshow('image', image)

            (H, W) = image.shape[:2]
            size_shape = (W, H)

            # for video creation
            img_array.append(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # write video
    print("Saving video...")
    out = cv2.VideoWriter(str(Path(video_dst_dir).joinpath(video_name + '_result.avi')),
                          cv2.VideoWriter_fourcc(*'DIVX'), 16,
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

    video_name = 'case_1.mp4'

    best_checkpoint = os.path.join('denred0_model', 'best_checkpoint',
                                   'senet154_hands_epoch=8_val_loss=0.026_val_acc=0.989_val_f1_epoch=0.989.ckpt')
    image_size_model = 224

    inference(video_src_dir=video_src_dir, video_dst_dir=video_dst_dir, video_name=video_name,
              checkpoint=best_checkpoint, image_size_model=image_size_model)
