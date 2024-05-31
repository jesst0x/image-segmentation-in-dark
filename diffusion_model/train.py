import argparse
import torch
import os
import json

from torch.utils.data import DataLoader
import kornia

from data.custom_dataset import ConditionalDiffusionDataset
from model.conditional_diffusion import ConditionalDiffusion


parser = argparse.ArgumentParser()
# Hyperparameters
parser.add_argument('--epoch', default='10', help='Number of epoch')
parser.add_argument('--lr', default='0.00005', help='Learning rate')
parser.add_argument('--weight_decay', default='0', help='L2 regularization')
parser.add_argument('--count', default='1000', help='Number of image to train')

# Directories
parser.add_argument('--checkpoint_epoch', default='2', help='Checkpoint epoch')
parser.add_argument('--weight_path', default='', help='If specified, load the saved weight from this file')
parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Directory to save the weigth')
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
    count = int(args.count)

    print("**********************")
    print(torch.cuda.is_available())
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("############Loading Datasets###########")
    transform = [
        kornia.augmentation.RandomCrop((256, 256)),
        kornia.augmentation.Resize((128, 128), antialias=True),
    ]
    train_dataset = ConditionalDiffusionDataset(input_img_dir, condition_img_dir, annotation_path, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
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
    loss_summary = []
    diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{0}.pth"))
    print("############Training###########")
    for epoch in range(num_epochs):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            (x, c), _ = batch
            loss = diffusion_model(x, c)
            if step % 100 == 0:
                print(f"Epoch {epoch + 1} Step {step}/{count} Loss: {loss.item()}")
                losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_summary.append(losses)
        if checkpoint_every != 0:
            if (epoch + 1) % checkpoint_every == 0:
                # Checkpoint
                diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{epoch + 1}.pth"))
                json.dump({"loss": loss_summary}, open(os.path.join(checkpoint_dir, "loss.json"), 'w'))
        
    # Save weight at the end of training for evaluation later
    diffusion_model.save_checkpoint(os.path.join(checkpoint_dir, f"model_path_{num_epochs}.pth"))