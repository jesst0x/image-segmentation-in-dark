import os
import argparse
import random
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../data/generated', help='Directory with raw dataset')
parser.add_argument('--output_dir', default ='../data/test/stylenat_256', help='Directory to save resize dataset')
parser.add_argument('--count', default ='3500', help='Number of Images')
parser.add_argument('--image_name', default ='stylenat', help='Image Name')
parser.add_argument('--image_size', default ='256', help='Output image size')

# Data preprocessing to create synthetic low light images 
if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.dataset_dir
    input_dir = args.input_dir
    data_dir = args.dataset_dir
    count = int(args.count)
    img_name = args.image_name
    img_size = int(args.image_size)
    

    darkened_summary = {}
    files = os.listdir(input_dir)
    for file in tqdm(files):
        darkened_img_tensor, gamma = adjust_gamma(os.path.join(input_dir, file))
        darkened_img = v2.functional.to_pil_image(darkened_img_tensor)
        darkened_img.save(os.path.join(output_dir, file))
        darkened_summary[file] = gamma
    json.dump(darkened_summary, open(os.path.join(output_dir, 'darkened_summary.json'), 'w'))