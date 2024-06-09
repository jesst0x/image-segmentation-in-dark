import sys
import argparse
import torch
import os
from tqdm import tqdm

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from torch_utils_lib.engine import evaluate
import torch_utils_lib.utils as torch_lib_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--count', default='10', help='Number of images to draw segmentation mask for analysis')
parser.add_argument('--batch_size', default='4', help='Batch size')

# Directories
parser.add_argument('--weight_path', default='./checkpoints/model_path_20.pth', help='If specified, load the saved weight from this file')
parser.add_argument('--normal_light_img_dir', default='../../coco_val500', help='Path to normal light images')
parser.add_argument('--low_light_img_dir', default='../../coco_val500_augmented', help='Path to low light image')
parser.add_argument('--annotation', default='../Dataset/annotations/instances_val2017_subset500.json', help='Path to COCO annotation json file')
parser.add_argument('--save_image_dir', default='outputs/epoch_20', help='Path to save images with predicted segmentation mask')

SUFFIXES = ["trained_original", "trained_synthetic", "pretrained_original", "pretrained_synthetic"]

if __name__ == '__main__':
    args = parser.parse_args()
    normal_image_dir = args.normal_light_img_dir
    low_light_image_dir = args.low_light_img_dir
    annotation = args.annotation
    weight_path = args.weight_path
    save_image_dir = args.save_image_dir
    count = int(args.count)
    batch_size = int(args.batch_size)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    print("################# Loading Data ###############")
    original_dataset, original_loader = utils.load_data(normal_image_dir, annotation, batch_size, False)
    augmented_dataset, augmented_loader = utils.load_data(low_light_image_dir, annotation, batch_size, False)
    
    # Mask R-CNN with ResNet50 backbone loaded with weight from checkpoint
    print("############ Loading Model###########")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False, trainable_backbone_layers=1)
    if weight_path != "":
        model.load_state_dict(torch.load(weight_path))
    model.to(device)
    
    
    # Evaluating the result of prediction, ie. AP
    print("############# Evaluating on Original Coco Dataset #############")
    evaluate(model, original_loader, device=device)
    print("############# Evaluating on Augmented Coco Dataset #############")
    evaluate(model, augmented_loader, device=device)
    
    
    # Compare with pre-trained models
    pretrained_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, progress=False)
    pretrained_model.to(device)


    print("##################### Draw Segmentation Mask ####################")
    indices = torch.randperm(len(original_dataset)).tolist()[:count]
    image_ids = []
    # Draw segmentation mask for analysis
    for i in tqdm(indices):
        # Natural image from Coco dataset and its synthetic low light image pair
        original_image, original_target = original_dataset[i]
        augmented_image, augmented_target = augmented_dataset[i]
        model.eval()
        pretrained_model.eval()
        
        with torch.no_grad():
            # Prediction from trained model
            original_img_pred = model(original_image.unsqueeze(0).to(device))[0]
            augmented_img_pred = model(augmented_image.unsqueeze(0).to(device))[0]
            # Prediction from pre-trained mask rcnn model
            pretrained_original_img_pred = pretrained_model(original_image.unsqueeze(0).to(device))[0]
            pretrained_augmented_img_pred = pretrained_model(augmented_image.unsqueeze(0).to(device))[0]
        
        # Draw predicted mask from trained and pre-trained models
        masked_original_img = utils.draw_mask(original_image, original_img_pred)
        masked_augmented_img = utils.draw_mask(augmented_image, augmented_img_pred)
        masked_pretrained_original_img = utils.draw_mask(original_image, pretrained_original_img_pred)
        masked_pretrained_augmented_img = utils.draw_mask(augmented_image, pretrained_augmented_img_pred)
        
        # Save images
        image_id = str(original_target["image_id"]).rjust(12, "0")
        image_ids.append(image_id + ".jpg")
        images = [masked_original_img,masked_augmented_img,masked_pretrained_original_img, masked_pretrained_augmented_img]
        utils.save_images(images, SUFFIXES, save_image_dir, image_id)
            
    print(image_ids)
            