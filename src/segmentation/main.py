import torch
import numpy as np
import albumentations as A
from torchvision.transforms import ToTensor
from .net import UNet

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = (512, 512)


class UNetInference(torch.nn.Module):

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
    ) -> None:

        super().__init__()

        self.device = device

        self.base_tf = A.Compose([
            A.Resize(
                width=IMG_SIZE[0],
                height=IMG_SIZE[1],
                p=1.,
            ),
            A.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ])

        self.to_tensor = ToTensor()

        self.net = UNet(num_classes)
        self.net.to(device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net.forward(x)
    
    @torch.no_grad()
    def inference(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        transformed = self.base_tf(
            image=frame,
        )

        frame = transformed['image']

        frame = self.to_tensor(frame).to(self.device).unsqueeze(0)
        pred = self.net.forward(frame).squeeze(0)
        mask = torch.argmax(pred, dim=0).cpu().numpy().astype(np.uint8) * 255

        return mask