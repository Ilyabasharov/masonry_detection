#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import tqdm
import torch
import click
import numpy as np

from segmentation import UNetInference
from masonry import MasonryShapeEvaluation

image_ext = ['jpg', 'jpeg', 'png', 'webp', ]
video_ext = ['mp4', 'mov', 'avi', 'mkv', ]

class MasonryDetection:

    def __init__(
        self,
        brick_width: float,
        brick_height: float,
        threshold_shape: float,
        device: str='cpu',
    ) -> None:

        self.shape = MasonryShapeEvaluation(
            real_shape_brick=(brick_width, brick_height),
            threshold_shape=threshold_shape,
        )

        self.net = UNetInference(
            num_classes=2,
            device=device,
        )

    def forward(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        mask = self.net.inference(image)
        mask = (~mask.astype(bool)).astype(int)

        image = self.shape.forward(
            image=image,
            mask=mask,
        )

        return image


@click.command()
@click.option(
    '--brick_width',
    default=250,
    help='Brick width in millimeters.',
)
@click.option(
    '--brick_height',
    default=250,
    help='Brick height in millimeters.',
)
@click.option(
    '--threshold_shape',
    default=10,
    help='Threshold masonry shape in millimeters.',
)
@click.option(
    '--input_images_path',
    default='../data/PPKE-SZTAKI-MasonryBenchmark/train/images',
    help='Input images path.',
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
    brick_width: float,
    brick_height: float,
    threshold_shape: float,
    input_images_path: str,
    annotate: bool,
    where_to_save: str,
) -> None:

    if annotate:
        os.makedirs(
            name=where_to_save,
            exist_ok=True,
        )

    for path in (input_images_path, ):
        assert os.path.exists(path), f'{path} does not exist!'

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

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    model = MasonryDetection(
        brick_height=brick_height,
        brick_width=brick_width,
        threshold_shape=threshold_shape,
        device=device,
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

        image = model.forward(frame)

        if annotate:
            cv2.imwrite(
                filename=os.path.join(where_to_save, f'{counter}.png'),
                img=image[..., ::-1],
            )

        counter += 1
        progress_bar.update()

if __name__ == '__main__':
    main()