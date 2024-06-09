# Low Light Instance Segmentation
This is the code implementation of the project **Low Light Instance Segmentation** that investigates generalizability of instance segmentation state-of-model to low light images, and explores other techniques like Diffusion Model to improve the performance. Github repo is [here](https://github.com/jesst0x/image-segmentation-in-dark).

## Requirement

Setting up a virtual environment as following is recommended:-

```
conda env create -n llis python=3.10
conda activate llis
pip install -r requirements.txt
```

This implementation is in Pytorch. Please install Pytorch library by following official [link](https://pytorch.org/get-started/locally/).

We use Pytorch reference util library to interact with COCO dataset, you can download it from following:-

```
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

```

This is saved in `./mask_rcnn/torch_utils_lib`.

## Data Preparation
1. You can download source images of COCO datasets used in the project from the official site,  [COCO](https://cocodataset.org/#download)
2. We only use subset of 5000 training, 500 validation and 500 test images. To reproduce the result, please use the annotation files in folder `./Dataset/annotations` downloadable with this repo. Otherwise, please download annotation files from [COCO](https://cocodataset.org/#download) if intend to train with larger dataset.

3. To build a pipeline of synthetic low light images using COCO dataset, please run the following commands:-

```
python ./Dataset/data_preprocessing.py --input_dir <path/to/dataset_input> 
```

The images are augmented by Gamma correction and Gaussian noise addition to produce synthetic low light images. Output images are saved in new folder with suffix '_augmented` in the same parent directory as input dataset.


## Training

### Fine-tuning Mask RCNN
To train the model, run the following command from `./mask_rcnn` folder:-

```
python train.py --image_dir <path/to/synthetic_img_training_set> --annotation <path/to/annotation_file> --checkpoint_dir <path/to/save_checkpoint_weight> --weight_path <path/to/load_previous_checkpoint>
```

There are additional hyperparameter arguments

- `batch_size`: Batch size of dataset 
- `epoch`: Number of epoch to run algorithm
- `weight_decay`: L2 regularization
- `learning_rate`: Initial learning rate
- `lr_step_size`: Learning rate step scheduler step size to decay the learning rate
- `lr_gamma`: Learning rate decay multiplication factor
- `trainable_backbone_layers`: ResNet50 backbone last number of trainable layers


### Conditional Diffusion Model
There are two input datasets to conditional diffusion model, ie. COCO dataset as *x<sub>0</sub>* and synthetic low light image as condition *c*.

Run the following command from folder `./diffusion_model` to train model to generate restored version from low-light images:-

```
python train.py --input_image_dir <path/to/coco_training_set> --conditional_image_dir <path/to/synthetic_low_light_images> --annotation <path/to/annotation_file> --checkpoint_dir <path/to/save_checkpoint_weight> --weight_path <path/to/load_previous_checkpoint>
```

There are additional hyperparameter arguments:-

- `batch_size`: Batch size of dataset 
- `epoch`: Number of epoch to run algorithm
- `weight_decay`: L2 regularization
- `learning_rate`: Initial learning rate
- `lr_step_size`: Learning rate step scheduler step size to decay the learning rate
- `lr_gamma`: Learning rate decay multiplication factor

There other arguments to monitor training:-
- `sample_every`: Backward sampling for every specified epoch to track generated images
- `checkpoint_epoch`: Save model weights for every specified epoch

We adopt two stages models for using Conditional Diffusion Model for image restoration followed by Mask R-CNN for instance segmentation. After training Diffusion Model, the output images are used for continuing training with Mask R-CNN.

## Sampling / Inference 

To generate restored images from Conditional Diffusion Models, run the following command from folder `./diffusion_model`:-

```
python inference.py --input_image_dir <path/to/normal_light_images> --conditional_image_dir <path/to/synthetic_low_light_images> --output_image_dir <path/to/save_output_images> --annotation <path/to/annotation_file> --weight_path <path/to/load_model_weight> --batch_size 4
```

Only images in specified argument `--conditional_image_dir` is used for inference to generate restore images. "Normal light" images specified in argument `-input_image_dir` is only used to plotting images for comparison.


## Evaluation

To evaluate the performance of trained model on test set, run the following command from `./mask_rcnn` folder:-

```
python evaluate.py --normal_light_img_dir <path/to/normal_light_images> --low_light_img_dir <path/to/low_light_images> --save_image_dir <path/to/save_images_with_drawn_mask> --weight_path <path/to/load_model_weight> --annotation <path/to/ground_truth_annotation> --batch_size 4
```

This outputs AP values. Images with drawn predicted segmentation mask are saved in folder specified in argument `--save_image_dir`.

Optional arguments
- `count`: Number of images to draw segmentation mask for qualitative analysis

## Other
Folder `./notebook` contains notebooks files used for fast experiments and visualization of small dataset. Full implementation of models are in other folders respective model names.


## Acknowledgement

The Unet Architecure used in diffusion model is based on [Tensorflow official implementation](https://github.com/hojonathanho/diffusion) and its [unofficial Pytorch adaption](https://github.com/lucidrains/denoising-diffusion-pytorch) with our modification. 

ConvNeXT block is based on the [official Pytorch implementation](https://github.com/facebookresearch/ConvNeXt) of the [A ConvNet for the 2020s paper](https://arxiv.org/pdf/2201.03545).

Pytorch reference tool [library](https://github.com/pytorch/vision) is used to interact with COCO dataset and Mask R-CNN model.

Dataset for training is from [COCO dataset](https://cocodataset.org/#home).
