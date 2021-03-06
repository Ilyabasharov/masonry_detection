#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import PIL
import click
import tqdm
import numpy as np
from masonry import MasonryShapeEvaluation

image_ext = ['jpg', 'jpeg', 'png', 'webp', ]

@click.command()
@click.option(
    '--input_images_path',
    default='../data/PPKE-SZTAKI-MasonryBenchmark/train/images',
    help='Input images path.',
)
@click.option(
    '--input_masks_path',
    default='../data/PPKE-SZTAKI-MasonryBenchmark/train/labels',
    help='Input masks path.',
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
    input_masks_path: str,
    annotate: bool,
    where_to_save: str,
) -> None:

    if annotate:
        os.makedirs(
            name=where_to_save,
            exist_ok=True,
        )
    
    image_names, masks_names = [], []

    for path, names in zip((input_images_path, input_masks_path), (image_names, masks_names)):
        assert os.path.exists(path), f'{path} does not exist!'
        assert os.path.isdir(path), f'{path} is not a directory!'

        ls = os.listdir(path)

        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                names.append(os.path.join(path, file_name))
        
    assert len(image_names) == len(masks_names), f'Number of images and masks should be the same.'

    model = MasonryShapeEvaluation()

    counter = 0
    progress_bar = tqdm.tqdm(total=len(image_names))

    while True:

        ret = False
        
        if counter < len(image_names):

            mask = (~np.asarray(PIL.Image.open(masks_names[counter])).astype(bool)).astype(int)
            image = np.asarray(PIL.Image.open(image_names[counter]))

            ret = True

        if not ret:
            break

        image = model.forward(
            image=image,
            mask=mask,
        )

        if annotate:
            cv2.imwrite(
                filename=os.path.join(where_to_save, f'{counter}.png'),
                img=image[..., ::-1],
            )

        counter += 1
        progress_bar.update()

if __name__ == '__main__':
    main()