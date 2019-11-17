import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torchvision.transforms as transforms
from numpy.linalg import norm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

def cos_sim(a,b):
    return 1 - dot(a, b)/(norm(a)*norm(b))

def euc_dis(a,b):
    return norm(a-b)

class FaceIdentityLoss(nn.Module):
    def __init__(self, model):
        super(FaceIdentityLoss, self).__init__()
        self.model = model

    def face_identity_loss(self, img1, img2): 
        img1_emb = self.model(img1.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        img2_emb = self.model(img2.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        return cos_sim(img1_emb, img2_emb)

    def batch_face_identity_loss(self, batch_img1, batch_img2):
        batch_size = batch_img1.shape[0]
        loss = 0
        for index in range(batch_size):
            loss += self.face_identity_loss(batch_img1[index], batch_img2[index])
        return loss/batch_size

    def forward(self, out_images, target_images):
        identity_loss = self.batch_face_identity_loss(out_images, target_images) 
        return identity_loss

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
