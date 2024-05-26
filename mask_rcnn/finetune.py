import sys
sys.path.append('../../mask_rcnn')

import argparse
import torch
import os
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader

from torch_utils_lib.engine import train_one_epoch
import torch_utils_lib.utils as torch_lib_utils

import utils

parser = argparse.ArgumentParser()
# Hyperparameters
parser.add_argument('--epoch', default='2', help='Number of epoch')
parser.add_argument('--lr', default='0.00005', help='Learning rate')
parser.add_argument('--weight_decay', default='0', help='L2 regularization')

# Directories
parser.add_argument('--checkpoint_epoch', default='0', help='Checkpoint epoch')
parser.add_argument('--weight_path', default='', help='If specified, load the saved weight from this file')
parser.add_argument('--checkpoint_dir', default='.', help='Directory to save the weigth')


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    weight_path = args.weight_path
    checkpoint_every = int(args.checkpoint_epoch)
    # Hyperparameters
    num_epochs = int(args.epoch)
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)
    # torch.cuda.empty_cache()
    print("**********************")
    print(torch.cuda.is_available())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("############Loading Datasets###########")
    train_dataset = utils.load_train_data()

    # indices = torch.randperm(len(train_dataset)).tolist()
    # subset = torch.utils.data.Subset(train_dataset, indices[:200])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=torch_lib_utils.collate_fn,
    )
    
    # val_augmented_dataset, val_original_dataset = utils.load_val_data()
    # val_augmented_dataloader = DataLoader(
    #     val_augmented_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     collate_fn=torch_lib_utils.collate_fn,
    # )
    # val_original_dataloader = DataLoader(
    #     val_original_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     collate_fn=torch_lib_utils.collate_fn,
    # )
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
        step_size=3,
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
