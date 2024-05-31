import os
from PIL import Image
import json

import kornia
from torch.utils.data import Dataset
import torchvision

class ConditionalDiffusionDataset(Dataset):
  """
  Dataset for conditional diffusion model.
  Args:
    natural_img_dir (string): Root directory of dataset as input to be noised in diffusion model.
    conditinal_img_dir (string): Root directory of images used as condition in diffusion model.
    anno_path (string): Path to the annotation json file, consisting of image file name.
    transform (callable, optional): Optional transform to be applied
      on input_image
  
  """
  def __init__(self, input_img_dir, conditional_img_dir, anno_path, transform=None):
    self.natural_img_dir = input_img_dir
    self.conditional_img_dir = conditional_img_dir
    with open(anno_path) as f:
      self.img_names = json.load(f)["filename"]
    self.transform = None
    if transform is not None:
        # Ensure that the random transform is the same for pair of input and conditional
        # images.
        self.transform=kornia.augmentation.container.AugmentationSequential(
            *transform,
            data_keys=["input", "input"],
            same_on_batch=False
        )

  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, index: int):
    """
    Args:
      index (int): Index of the dataset
    Returns:
      tuple: Pair of natural and conditional images in (N, H, W) of value [0, 1], and image file name
    """
    if not isinstance(index, int):
      raise ValueError(f"Index must be of type integer, got {type(index)} instead.")
    natural_img = torchvision.io.read_image(os.path.join(self.natural_img_dir, self.img_names[index])) / 255
    conditional_img = torchvision.io.read_image(os.path.join(self.conditional_img_dir, self.img_names[index])) / 255
    if self.transform is not None:
        out = self.transform(natural_img,conditional_img)
        natural_img = out[0][0]
        conditional_img = out[1][0]
    img_name = self.img_names[index]
    return (natural_img, conditional_img), img_name