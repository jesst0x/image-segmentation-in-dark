import argparse
import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms import v2
import torchvision.utils
import torch
import numpy as np
import random
import json
from skimage.util import random_noise
from torcheval.metrics.functional import peak_signal_noise_ratio

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../../coco_train5000', help='Directory with raw dataset')

def adjust_gamma(image_path, gamma=[5, 10]):
  img = Image.open(image_path)
  img_tensor = v2.functional.pil_to_tensor(img)
  gamma = np.random.uniform(gamma[0], gamma[1])
  transform_img = v2.functional.adjust_gamma(img_tensor, gamma=gamma)
  return transform_img, gamma

def add_gaussian_noise(image_path, var_range=[0.1, 0.5]):
  img = Image.open(image_path)
  img_tensor = v2.functional.pil_to_tensor(img)
  var = np.random.uniform(var_range[0], var_range[1])
  # Normalize to [-1, 1]
  img_normalized = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_tensor / 255.0)
  noisy_image = random_noise(img_normalized, var=var, clip=True)
  noisy_image = torch.tensor(noisy_image * 0.5 + 0.5)
  return noisy_image

# Data preprocessing to create synthetic low light images 
if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input_dir 

    random.seed(23)
    
    darkened_dir = input_dir  + '_darkened'
    darkened_summary = {}
    files = os.listdir(input_dir)
    print("########Darken images#####")
    for file in tqdm(files):
        darkened_img_tensor, gamma = adjust_gamma(os.path.join(input_dir, file), [3, 8])
        darkened_img = v2.functional.to_pil_image(darkened_img_tensor)
        darkened_img.save(os.path.join(darkened_dir, file))
        darkened_summary[file] = gamma
    json.dump(darkened_summary, open('coco_darkened_summary_train_500.json', 'w'))
    
    print("#####Add Gaussian Noise#####")
    augmented_dir = input_dir  + '_augmented'
    files = os.listdir(darkened_dir)
    for f in tqdm(files):   
        noisy_img = add_gaussian_noise(os.path.join(darkened_dir, f), [0.001, 0.01])
        torchvision.utils.save_image(noisy_img, os.path.join(augmented_dir, f))
        
    print("####Calculate SNR#####")
    augmented_snr = []
    files = os.listdir(augmented_dir)
    grey = []
    for f in tqdm(files):
        normal_img_path = os.path.join(input_dir, f)
        low_light_img_path = os.path.join(augmented_dir, f)
        normal_tensor = v2.functional.pil_to_tensor(Image.open(low_light_img_path)) / 255
        low_light_tensor = v2.functional.pil_to_tensor(Image.open(normal_img_path)) / 255
        if low_light_tensor.shape != normal_tensor.shape:
            grey.append(f)
            continue
        snr = peak_signal_noise_ratio(low_light_tensor, normal_tensor)
        if snr < 1:
          print(snr, f)
        augmented_snr.append(snr)
      
    snr_summary = {'mean': float(np.mean(augmented_snr).squeeze()), 'max': float(max(augmented_snr)), 'min': float(min(augmented_snr)), 'std': float(np.std(augmented_snr).squeeze())}
    print(snr_summary)
    json.dump(snr_summary, open('coco_augmented_val_train_snr.json', 'w'))
        