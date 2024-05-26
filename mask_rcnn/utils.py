from torchvision.transforms import ToTensor
from torchvision import datasets

### Paths to dataset.
TRAINING_IMAGES_PATH = "/home/jesstoh/coco_train5000_augmented"
VAL_ORGINAL_IMAGES_PATH = "/home/jesstoh/coco_val500"
VAL_AUGMENTED_IMAGES_PATH = "/home/jesstoh/coco_val500_augmented"
TRAINING_ANNOTATIONS_PATH = "/home/jesstoh/image-segmentation-in-dark/Dataset/annotations/instances_train2017_subset5000.json"
VAL_ANNOTATIONS_PATH = "/home/jesstoh/image-segmentation-in-dark/Dataset/annotations/instances_val2017_subset500.json"

def load_train_data():
    train_dataset = datasets.CocoDetection(TRAINING_IMAGES_PATH, TRAINING_ANNOTATIONS_PATH, transform=ToTensor())
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=("all")) 
    return train_dataset 

def load_val_data():
    val_augmented_dataset = datasets.CocoDetection(VAL_AUGMENTED_IMAGES_PATH, VAL_ANNOTATIONS_PATH, transform=ToTensor())
    val_augmented_dataset = datasets.wrap_dataset_for_transforms_v2(val_augmented_dataset, target_keys=("all"))
    
    val_original_dataset = datasets.CocoDetection(VAL_ORGINAL_IMAGES_PATH, VAL_ANNOTATIONS_PATH, transform=ToTensor())
    val_original_dataset = datasets.wrap_dataset_for_transforms_v2(val_original_dataset, target_keys=("all"))
    return val_augmented_dataset, val_original_dataset 