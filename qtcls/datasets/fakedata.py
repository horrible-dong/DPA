# ---------------------------------------
# Modified from torchvision by QIU Tian
# ---------------------------------------

__all__ = ['FakeData']

from typing import Any, Tuple

import torch
from torchvision.transforms import ToPILImage

from ._base_ import BaseDataset


class FakeData(BaseDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images.

    Args:
        size (int): Size of the dataset. Default: 1000 images
        split (str): The dataset split. E.g. ``train``, ``val``, ``test``...
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 1000
        random_offset (int): Offsets the index-based random seed used to generate each image. Default: 0
        transform (callable, optional): A function/transform that takes in an PIL image and transforms it.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        batch_transform (callable, optional): A function/transform that takes in a batch and transforms it.

    """

    def __init__(self, size: int, split: str = 'train', image_size: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000, random_offset: int = 0, transform=None, target_transform=None):
        super().__init__('(no root required)', split, transform, target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        image = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0].item()
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        image = ToPILImage()(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return self.size
