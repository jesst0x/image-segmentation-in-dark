import os
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import torch_utils_lib.utils as torch_lib_utils
from torchvision.utils import draw_segmentation_masks, save_image

def load_data(image_dir, annotation_path, batch_size=4, shuffle=True, seed=23):
    """
    Load images and targets into dataset and dataloader.

    Inputs:
    - image_dir: Directory path consisting of input images
    - annotation_path: File path of annotation json file consisting of ground truth labels
    - batch_size: Batch size to use for output dataset and dataloader
    - shuffle: Whether to shuffle the data
    
    Returns a tuple of (dataset, dataloader)
    """
    torch.manual_seed(seed)
    dataset = datasets.CocoDetection(image_dir, annotation_path, transform=ToTensor())
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("all"))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=torch_lib_utils.collate_fn,
    )
    return dataset, data_loader

def compute_loss(model, dataloader):
    """
    Compute the loss 

    Inputs:
    - model: Model with preloaded trained weight for evaluation
    - dataloader: Dataloader iterating through dataset batches. Each data in batch consists of tuple of (image_tensor, ground_truth_label)
    
    Returns:
    - loss: Dict of various loss.
    """
    losses = defaultdict(int)
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        for name, val in loss_dict.items():
            losses[name] += val.item()
        
    total_loss = sum(loss for loss in losses.values())
    losses["loss"] = total_loss
    return losses
      
def draw_mask(image, pred, threshold=0.7):
    """
    Draw predicted segmentation mask on the input image

    Inputs:
    - image: Input image tensor of shape (C, H, W)
    - pred: Prediction from model
    - threshold: Probability threshold to plot the predicted mask
    
    Returns:
    - output_image: Image tensor with segmentatin mask of shape (C, H, W)
    """
    masks = (pred["masks"] > threshold).squeeze(1)
    input_image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    # Segmentation mask on top of the the image
    output_img = draw_segmentation_masks(input_image, masks, alpha=0.7)
    
    return output_img

def save_images(images, suffixes, save_dir, image_id):
    """
    Helper method to conventiently save the image with different suffixes on the file name.
    Inputs:
    - image: List of image tensor of shape (C, H, W) and value of range [0, 255]
    - suffixes: List of suffix to append on image file name
    - save_dir: Directory to save the images
    - image_id: Image id from original COCO dataset.
    """
    for img, suffix in zip(images, suffixes):
        save_image(img / 255, os.path.join(save_dir, f"{image_id}_{suffix}.jpg"))