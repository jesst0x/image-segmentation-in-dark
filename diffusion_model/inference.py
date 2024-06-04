import sys
sys.path.append('./model')
from tqdm import tqdm
import argparse
import torch
import os
import time
import gc

import kornia
from torch.utils.data import DataLoader
from data.custom_dataset import ConditionalDiffusionDataset
from model.conditional_diffusion import ConditionalDiffusion

import util

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default='4', help='Batch size')
parser.add_argument('--inference_step', default='1000', help='Number of steps to sample for DDIM')
parser.add_argument('--weight_path', default='./checkpoints/128_128_linear_darkened/model_path_520.pth', help='If specified, load the saved weight from this file')
parser.add_argument('--input_image_dir', default='../../lis_normal_test500_rgb', help='Path to natural light image for training')
parser.add_argument('--conditional_img_dir', default='../../lis_low_light_test500_rgb', help='Path to synthetic low light image as condition')
parser.add_argument('--output_img_dir', default='./data/lis_test500_pred_2', help='Path to save model output image')
parser.add_argument('--annotation', default='../Dataset/annotations/lis_coco_jpg_test500_ids_rgb.json', help='Path to file list')


if __name__ == '__main__':
    args = parser.parse_args()
    weight_path = args.weight_path
    input_img_dir = args.input_image_dir
    condition_img_dir = args.conditional_img_dir
    output_img_dir = args.output_img_dir
    batch_size = int(args.batch_size)
    inference_step = int(args.inference_step)
    annotation_path = args.annotation
    # Sampling to save as image
    save_steps = [i for i in range(0, 1000, 50)]
    save_steps.append(999)
    if not os.path.exists(output_img_dir):
      os.makedirs(output_img_dir)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Resizing due to long inference time
    transform = [
        kornia.augmentation.Resize((400, 600), antialias=True),
    ]
    
    print("############Loading Datasets###########")
    dataset = ConditionalDiffusionDataset(input_img_dir, condition_img_dir, annotation_path, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    
    print("############Loading Model###########")
    model = ConditionalDiffusion(device=device)
    model.load_checkpoint(weight_path)
    model.eval()
    
    start = time.time()
    print("############Predicting###########")
    for i, batch in enumerate(data_loader):
        with torch.no_grad():
          (x, c), ids = batch
          generated = model.p_sample_progressive(c, save_steps)
          # Save each output image at timestep 0
          util.save_images(generated[-1], ids, output_img_dir)
          # Save progressive image sampling at save_steps
          util.plot(generated, x, c, True, os.path.join(output_img_dir, f"batch_{i}.png",))
          duration = time.time() - start
          print(f"Time elapsed: {util.format_duration(duration)}")
          del generated
          gc.collect()
