import numpy as np

from .utils import (
    fit_axial_segments,
    get_pix_dims_brick,
    plot_contours,
)

class MasonryShapeEvaluation:

    def __init__(
        self,
        real_shape_brick: tuple=(250., 120.), #width height mm
        threshold_shape: float=10, # mm
    ) -> None:

        self.real_shape_brick = real_shape_brick
        self.threshold_shape = threshold_shape

    def forward(
        self,
        image: np.ndarray,
        mask: np.ndarray, #0 - brick, 1 - masonry
    ) -> np.ndarray:

        contours = fit_axial_segments(
            mask=mask[..., None],
            rtol=0.03,
            atol=5,
        )

        pix_shape = get_pix_dims_brick(
            mask=mask,
        )

        image = plot_contours(
            image=image,
            contours=contours,
            pix_to_real=(
                self.real_shape_brick[0] / pix_shape[0],
                self.real_shape_brick[1] / pix_shape[1],
            ),
            treshold_real=self.threshold_shape,
        )

        return image