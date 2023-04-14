from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import numpy as np
from torch import nn, Tensor
from torchvision import ops

# Randomly gamma correct image
class RandomGammaCorrection(nn.Module):
    """
    Apply Gamma Correction to the images
    """
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        #sample log uniformly in range [log(gamma_range[0]), log(gamma_range[1])]
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        return F.adjust_gamma(img, gamma)

    def forward(self, img):
        return self.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(gamma_range={0})'.format(self.gamma_range)

#Random scale jitter
class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.
    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 1.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target

if __name__ == '__main__':


    #Test the augmentations
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    #Load image
    img = Image.open('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/Serier/Series2/CellsCorr/Serie_2_ImageCorr_-1_3992_PC_Cell_Row1_Col_1.png')

    #Convert to tensor
    img_org = F.to_tensor(img)

    #Create augmentations
    aug = ScaleJitter((512, 512))
    aug2 = RandomGammaCorrection()

    #Apply augmentations
    img_scale, _ = aug(img_org, None)
    img_gamma = aug2(img_org)

    #Convert to pil image
    img_scale = F.to_pil_image(img_scale)
    img_gamma = F.to_pil_image(img_gamma)
