import torch
from torch import nn
from torch.nn import functional as F

class DoubleBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        middle_channels: int=None,
        dropout: bool=False,
        p: float=0.2,
    ) -> None:

        super().__init__()

        if middle_channels is None:
            middle_channels = out_channels

        self.first_step = nn.Sequential(
            nn.Dropout(p) if dropout else nn.Identity(),
            nn.Conv2d(
                in_channels,
                middle_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )
        self.second_step = nn.Sequential(
            nn.Conv2d(
                middle_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.first_step(x)
        x = self.second_step(x)

        return x

class Encoder(nn.Module):

    def __init__(
        self,
        channels: list,
        max_channels: int,
    ) -> None:

        super().__init__()
        
        last_or_not = lambda i, length: True if i == length - 1 else False

        self.blocks = nn.ModuleList([
            DoubleBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                middle_channels=max_channels if last_or_not(i, len(channels)) else channels[i + 1]
            )
            for i in range(len(channels) - 1)
        ])

        self.downsample = nn.MaxPool2d(
            kernel_size=(2, 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple:

        features = []

        for block in self.blocks:

            x = block(x)
            features.append(x)
            x = self.downsample(x)

        return features

class Decoder(nn.Module):

    def __init__(
        self,
        channels: list,
        p: float=0.2,
    ) -> None:

        super().__init__()

        self.channels = channels

        self.blocks = nn.ModuleList([
            DoubleBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                dropout=True,
                p=p,
            )
            for i in range(len(self.channels) - 1)
        ])

        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=(2, 2),
                stride=(2, 2),
            )
            for i in range(len(self.channels) - 1)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        
        for i in range(len(self.channels) - 1):

            x = self.upsample[i](x)

            _, _, H, W = x.shape
            _, _, H_feat, W_feat = features[i].shape
            
            if H != H_feat or W != W_feat:

                features[i] = torch.nn.functional.interpolate(
                    input=features[i],
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True,
                )

            x = torch.cat(
                tensors=[x, features[i]],
                dim=1,
            )

            x = self.blocks[i](x)

        return x


class UNet(nn.Module):
    """

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(
        self, 
        num_classes,
        min_channels: int=32,
        max_channels: int=512, 
        num_down_blocks: int=5,
        input_channels: int=3,
    ) -> None:

        super(UNet, self).__init__()
        
        self.num_classes = num_classes

        channels = self.make_channels(
            min_channels=min_channels,
            num_down_blocks=num_down_blocks,
        )
        
        self.decoder = Decoder(
            channels=channels[::-1],
        )
        self.encoder = Encoder(
            channels=[input_channels, ] + channels,
            max_channels=max_channels,
        )
        self.head = nn.Conv2d(
            in_channels=channels[0],
            out_channels=num_classes,
            kernel_size=(1, 1),
        )

        self.init_weights()

    @staticmethod
    def make_channels(
        min_channels: int,
        num_down_blocks: int,
    ) -> list:

        output = [
            min_channels*(2**i)
            for i in range(num_down_blocks)
        ]

        return output

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        B, C, H, W = x.shape
        
        features = self.encoder(x)[::-1]

        x, last = features[0], features[1:]

        out = self.decoder(
            x=x,
            features=last,
        )

        logits = self.head(out)
        
        _, _, H_feat, W_feat = logits.shape
            
        if H != H_feat or W != W_feat:

            logits = torch.nn.functional.interpolate(
                input=logits,
                size=(H, W),
                mode='bilinear',
                align_corners=True,
            )

        assert logits.shape == (B, self.num_classes, H, W), \
            f'Wrong shape of the logits. Got: {logits.shape}, expected: {(B, self.num_classes, H, W)}'
        
        return logits

    def init_weights(
        self,
    ) -> None:

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)