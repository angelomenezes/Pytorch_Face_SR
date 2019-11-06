# Data loader for Super-Resolution model 

#import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import torch
from os import listdir
from os.path import join
from PIL import Image
import numpy as np

#def load_image(path):
#    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    #return img[...,::-1]
#    return img

def load_image(path):
    img = Image.open(path)
    return img

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(224),
        CenterCrop(224),
        ToTensor()
    ])

class DatasetSuperRes(Dataset):
    def __init__(self, image_dir, target_dir, transform=None):        
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir)]

    def __getitem__(self, index):
        img_input = np.array(load_image(self.image_filenames[index]))
        target = np.array(load_image(self.target_filenames[index]))
        
        #img_input = np.reshape(img_input, (3, img_input.shape[0], target.shape[1]))
        #target = np.reshape(target, (3, target.shape[0], target.shape[1]))
        
        img_input = torch.from_numpy(img_input).float()
        target = torch.from_numpy(target).float()
        #img_input = torch.from_numpy(img_input.transpose(2, 0, 1)).float()
        #target = torch.from_numpy(target.transpose(2, 0, 1)).float()
        
        img_input = img_input.view(3,img_input.shape[0],img_input.shape[1])
        target = target.view(3,target.shape[0],target.shape[1])
        
        img_input = img_input / 255.0
        target = target / 255.0
        
        return img_input, target

    def __len__(self):
        return len(self.image_filenames)
