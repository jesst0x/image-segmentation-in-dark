import sys
sys.path.append('./model')

import argparse
import torch
import os
import json
import time

from torch.utils.data import DataLoader
import kornia

from data.custom_dataset import ConditionalDiffusionDataset
from model.conditional_diffusion import ConditionalDiffusion
import util

parser = argparse.ArgumentParser()
# Hyperparameters
parser.add_argument('--epoch', default='10000', help='Number of epoch')
parser.add_argument('--lr', default='0.00001', help='Learning rate')
parser.add_argument('--weight_decay', default='0', help='L2 regularization')
parser.add_argument('--batch_size', default='16', help='Batch size')

# Directories
parser.add_argument('--checkpoint_epoch', default='5', help='Checkpoint epoch')
parser.add_argument('--weight_path', default='./checkpoints/256_256_lr2e5/model_path_35.pth', help='If specified, load the saved weight from this file')
parser.add_argument('--checkpoint_dir', default='./checkpoints/256_256_lr2e5', help='Directory to save the weigth')
parser.add_argument('--input_image_dir', default='../../coco_train5000', help='Path to natural light image for training')
parser.add_argument('--conditional_img_dir', default='../../coco_train5000_augmented', help='Path to synthetic low light image as condition')
parser.add_argument('--annotation', default='../Dataset/annotations/coco_ids_train_5000.json', help='Path to file list')

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    weight_path = args.weight_path
    checkpoint_every = int(args.checkpoint_epoch)
    input_img_dir = args.input_image_dir
    condition_img_dir = args.conditional_img_dir
    annotation_path = args.annotation
    # Hyperparameters
    num_epochs = int(args.epoch)
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)
    batch_size = int(args.batch_size)

    print("**********************")
    print(torch.cuda.is_available())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("############Loading Datasets###########")
    transform = [
        kornia.augmentation.RandomCrop((256, 256)),
        # kornia.augmentation.Resize((128, 128), antialias=True),
    ]
    train_dataset = ConditionalDiffusionDataset(input_img_dir, condition_img_dir, annotation_path, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    print("############Datasets Loaded###########")
    
    # Conditional Diffusion Model
    print("############Building Model###########")
    diffusion_model = ConditionalDiffusion(device=device)
    # Load from checkpoint
    if weight_path != "":
        diffusion_model.load_checkpoint(weight_path)
    # Optimizer
    params = [p for p in diffusion_model.get_model_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    print(f"Number of model parameters: {util.get_parameter_count(diffusion_model)}")
    loss_summary = []
    # diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{0}.pth"))
    start = time.time()
    print("############Training###########")
    for epoch in range(35, 35 + num_epochs):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            (x, c), _ = batch
            x = x.to(device)
            c = c.to(device)
            loss = diffusion_model(x, c)
            if step % 10 == 0:
                print(f"Epoch {epoch + 1} Step {batch_size * step}/{5000} Loss: {loss.item()}")
                losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_summary.append(losses)
        if checkpoint_every != 0:
            if (epoch + 1) % checkpoint_every == 0:
                # Checkpoint
                diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{epoch + 1}.pth"))
                json.dump({"loss": loss_summary}, open(os.path.join(checkpoint_dir, "loss2.json"), 'w'))
        duration = time.time() - start
        print(f"Time elapsed: {util.format_duration(duration)}")
    # Save weight at the end of training for evaluation later
    diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{num_epochs}.pth"))