# Data loader for Super-Resolution model 

#import cv2
from torch.utils.data import Dataset
import torch
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

#def load_image(path):
#    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    #return img[...,::-1]
#    return img

def load_image(path):
    img = Image.open(path).convert('YCbCr')
    img = img.resize((224,224), Image.BICUBIC)
    y, _, _ = img.split()
    return y  

#def totensor():
#    return transforms.Compose([transforms.ToTensor(),])
                               #transforms.Normalize(0.5, 0.5),])

class DatasetSuperRes(Dataset):
    def __init__(self, image_dir, target_dir, transform=None):        
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        self.target_filenames = [join(target_dir, x) for x in listdir(target_dir)]

    def __getitem__(self, index):
        img_input = np.array(load_image(self.image_filenames[index]))
        target = np.array(load_image(self.target_filenames[index]))
       
        img_input = torch.from_numpy(img_input).float()
        target = torch.from_numpy(target).float()
        
        img_input = img_input.view(1,img_input.shape[0],img_input.shape[1])
        target = target.view(1,target.shape[0],target.shape[1])
        
        img_input = img_input / 255.0
        target = target / 255.0
        
        #img_input = totensor()(img_input)
        #target = totensor()(target)
        
        return img_input, target

    def __len__(self):
        return len(self.image_filenames)
