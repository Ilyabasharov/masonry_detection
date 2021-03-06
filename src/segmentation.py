#!/usr/bin/python3
# coding: utf-8

import os
import cv2
import click
import tqdm
import torch

from segmentation import UNetInference

image_ext = ['jpg', 'jpeg', 'png', 'webp', ]
video_ext = ['mp4', 'mov', 'avi', 'mkv', ]

@click.command()
@click.option(
    '--input_images_path',
    default='../data/PPKE-SZTAKI-MasonryBenchmark/train/images',
    help='Input images path.',
)
@click.option(
    '--path_to_weights',
    default='../weights/epoch=74-mean_iou=0.594.ckpt',
    help='Path_to_model_weights',
)
@click.option(
    '--annotate',
    is_flag=True,
    default=False,
    help='Save annotated files or not',
)
@click.option(
    '--where_to_save',
    default='../data/annotated',
    help='Where save annotated results.',
)
def main(
    input_images_path: str,
    path_to_weights: str,
    annotate: bool,
    where_to_save: str,
) -> None:
    
    for path in (path_to_weights, input_images_path):
        assert os.path.exists(path), f'{path} does not exist!'

    if annotate:
        os.makedirs(
            name=where_to_save,
            exist_ok=True,
        )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    if input_images_path[input_images_path.rfind('.') + 1:].lower() in video_ext:

        is_video = True
        capture = cv2.VideoCapture(input_images_path)
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    elif os.path.isdir(input_images_path):

        is_video = False
        image_names = []
        ls = os.listdir(input_images_path)

        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(input_images_path, file_name))
        
        length = len(image_names)

    else:

        raise NotImplementedError

    model = UNetInference(
        num_classes=2,
        device=device,
    )
    model.load_state_dict(
        torch.load(
            f=path_to_weights,
            map_location='cpu',
        )['state_dict']
    )

    counter = 0
    progress_bar = tqdm.tqdm(total=length)

    while True:
        
        ret = False

        if is_video:
            ret, frame = capture.read()
        else:
            if counter < len(image_names):
                frame = cv2.imread(image_names[counter])
                ret = True

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = model.inference(frame)

        if annotate:
            cv2.imwrite(
                filename=os.path.join(where_to_save, f'{counter}.png'),
                img=mask,
            )

        counter += 1
        progress_bar.update()

if __name__ == '__main__':
    main()