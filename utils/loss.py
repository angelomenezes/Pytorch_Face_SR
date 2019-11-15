import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torchvision.transforms as transforms
from numpy.linalg import norm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

#model_path = '/home/angelo/Desktop/Github/Feature-Extractors/'
model_path = '/home/jupyter/Feature-Extractors/'

os.sys.path.append(model_path)

def initialize_senet50_2048():
    import senet50_ft_pytorch.senet50_ft_dims_2048 as model
    network = model.senet50_ft(weights_path=model_path + 'senet50_ft_pytorch/senet50_ft_dims_2048.pth')
    network.eval()
    return network

def cos_sim(a,b):
    return 1 - dot(a, b)/(norm(a)*norm(b))

def euc_dis(a,b):
    return norm(a-b)

def return_activations(model, image):
    '''
    Currently it only accepts SeNet50 activations.
    '''
    activations = list()
    normalized_values = 0
    for i in range(2,7):
        f = model(image.view(1,3,image.shape[1],image.shape[2]))[i].detach().cpu().numpy()[:, :, 0, 0]
        normalized_values = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
        activations.append(normalized_values.reshape(-1))
    return np.array(activations)

def face_content_loss(model, img1, img2, metric='L2'):
    
    img1 = VGG2Face_normalization(img1)
    img2 = VGG2Face_normalization(img2)
    
    # Specific for SeNet50
    activations_img1 = return_activations(model, img1)
    activations_img2 = return_activations(model, img2)
    loss = 0
    
    # L1 Error or L2 Error or Euclidean Distance
    for layer in range(len(activations_img1)):
        if metric == 'L1':
            loss += np.sum(np.abs(activations_img1[layer] - activations_img2[layer]))
        elif metric == 'L2':
            loss += np.sum((activations_img1[layer] - activations_img2[layer])**2)
        else:
            loss += euc_dis(activations_img1[layer], activations_img2[layer])
    return loss

def batch_face_content_loss(model, batch_img1, batch_img2):
    batch_size = batch_img1.shape[0]
    loss = 0
    for index in range(batch_size):
        loss += face_content_loss(model, batch_img1[index],batch_img2[index])
    
    return loss/batch_size

def return_embedding(model, image):
    f = model(image.view(1,3,image.shape[1],image.shape[2]))[1].detach().cpu().numpy()[:, :, 0, 0]
    face_feats = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True)) # Normalizing embedding
    return face_feats.reshape(-1)

def face_identity_loss(model, img1, img2):
    img1 = VGG2Face_normalization(img1)
    img2 = VGG2Face_normalization(img2)
    
    img1_emb = return_embedding(model, img1)
    img2_emb = return_embedding(model, img2)
    return cos_sim(img1_emb, img2_emb)

def batch_face_identity_loss(model, batch_img1, batch_img2):
    batch_size = batch_img1.shape[0]
    loss = 0
    for index in range(batch_size):
        loss += face_identity_loss(model, batch_img1[index], batch_img2[index])
    
    return loss/batch_size

def VGG2Face_normalization(img):
    '''
    Normalization according to parameters of the used architecture
    '''
    VGG_mean = (131.0912, 103.8827, 91.4953)
    VGG_std = (1, 1, 1)
    return transforms.Normalize(VGG_mean,VGG_std)(img)

class FacePerceptionLoss(nn.Module):
    def __init__(self, model):
        super(FacePerceptionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.model = model
        

    def forward(self, out_images, target_images):
        
        # Scaling back the data from (0,1) to (0,255) since SEResNet did not use such standard
        out_images *= 255
        target_images *= 255
        
        # Identity Loss
        identity_loss = batch_face_identity_loss(self.model, out_images, target_images)
        # Content Loss
        content_loss = batch_face_content_loss(self.model, out_images, target_images)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss + 0.01 * identity_loss + 0.01 * content_loss

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
