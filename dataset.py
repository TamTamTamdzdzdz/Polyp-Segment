from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os
from PIL import Image
import numpy as np
from albumentations import (
    Compose,
    RandomRotate90,
    Flip,
    Transpose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomBrightnessContrast,
    HorizontalFlip,
    VerticalFlip,
    RandomGamma,
    RGBShift,
)
from torchvision.transforms import Resize, PILToTensor, ToPILImage, InterpolationMode
from torchvision.transforms import Compose as TC


# class UNetDataClass(Dataset):
#     def __init__(self, images_path, masks_path, transform):
#         super(UNetDataClass, self).__init__()
        
#         images_list = os.listdir(images_path)
#         masks_list = os.listdir(masks_path)
        
#         images_list = [images_path + image_name for image_name in images_list]
#         masks_list = [masks_path + mask_name for mask_name in masks_list]
        
#         self.images_list = images_list
#         self.masks_list = masks_list
#         self.transform = transform
        
#     def __getitem__(self, index):
#         img_path = self.images_list[index]
#         mask_path = self.masks_list[index]
        
#         # Open image and mask
#         data = Image.open(img_path)
#         label = Image.open(mask_path)
        
#         # Normalize
#         data = self.transform(data) / 255
#         label = self.transform(label) / 255
        
#         label = torch.where(label>0.65, 1.0, 0.0)
        
#         label[2, :, :] = 0.0001
#         label = torch.argmax(label, 0).type(torch.int64)
        
#         return data, label
    
#     def __len__(self):
#         return len(self.images_list)
    
class SegDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform=None, augmentation=None):
        super(SegDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)
        
        images_list = [os.path.join(images_path, image_name) for image_name in images_list]
        masks_list = [os.path.join(masks_path, mask_name) for mask_name in masks_list]
        
        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        # Augmentation
        if self.augmentation:
            augmented = self.augmentation(image=np.array(data), mask=np.array(label))
            data = Image.fromarray(augmented['image'])
            label = Image.fromarray(augmented['mask'])
        
        # Normalize
        data = self.transform(data) / 255
        label = self.transform(label) / 255
        
        label = torch.where(label > 0.65, 1.0, 0.0)
        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        
        return data, label
    
    def __len__(self):
        return len(self.images_list)



    
images_path = "data/train/train/"
masks_path =  "data/train_gt/train_gt/"
augmentation = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
    RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
])

transform = TC([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])
aug_dataset = SegDataClass(images_path, masks_path, transform=transform, augmentation=augmentation)
TRAIN_SIZE, TEST_SIZE = 0.9, 0.1
BATCH_SIZE = 16
train_aug_set, valid_aug_set = random_split(aug_dataset, 
                                    [int(TRAIN_SIZE * len(aug_dataset)) , 
                                     int(TEST_SIZE * len(aug_dataset))])
train_loader = DataLoader(train_aug_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_aug_set, batch_size=BATCH_SIZE, shuffle=False)