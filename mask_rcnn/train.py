import sys
sys.path.append('../../mask_rcnn')

import argparse
import torch
import os
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from torch_utils_lib.engine import train_one_epoch
import torch_utils_lib.utils as torch_lib_utils

import utils

parser = argparse.ArgumentParser()
# Hyperparameters
parser.add_argument('--epoch', default='30', help='Number of epoch')
parser.add_argument('--lr', default='0.0001', help='Learning rate')
parser.add_argument('--weight_decay', default='0', help='L2 regularization')

# Directories
parser.add_argument('--checkpoint_epoch', default='2', help='Checkpoint epoch')
parser.add_argument('--weight_path', default='', help='If specified, load the saved weight from this file')
parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Directory to save the weigth')
parser.add_argument('--image_dir', default='../../coco_train5000_augmented', help='Path to synthetic low light image for training')
parser.add_argument('--annotation', default='../Dataset/annotations/instances_train2017_subset5000.json', help='Path to COCO annotation json file')


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    weight_path = args.weight_path
    checkpoint_every = int(args.checkpoint_epoch)
    image_dir = args.image_dir
    annotation_path = args.annotation
    # Hyperparameters
    num_epochs = int(args.epoch)
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)

    print("**********************")
    print(torch.cuda.is_available())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("############Loading Datasets###########")
    train_dataset, train_dataloader = utils.load_data(image_dir, annotation_path, 4, True)
    print("############Datasets Loaded###########")
    
    # Mask R-CNN with ResNet50 backbone initialized with pre-trained weight.
    print("############Building Model###########")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False, trainable_backbone_layers=1)
    # Load from checkpoint
    if weight_path != "":
        model.load_state_dict(torch.load(weight_path))
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )
    
    print("############Training###########")
    # Fine tuning by continue training with 
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        
        if checkpoint_every != 0:
            if (epoch + 1) % checkpoint_every == 0:
                # Checkpoint
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_path_{epoch + 1}.pth"))
        # Update learning rate
        lr_scheduler.step()
        
    # Save weight at the end of training for evaluation later
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_path_{num_epochs}.pth"))