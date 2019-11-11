import torch
from torch import nn
from torchvision.models.vgg import vgg16
from numpy.linalg import norm
import numpy as np
model_path = '/home/angelo/Desktop/Github/Feature-Extractors/'
os.sys.path.append(model_path)
import senet50_128_pytorch.senet50_128 as model
    
def initialize_senet50():
    network = model.senet50_128(weights_path=model_path + 'senet50_128_pytorch/senet50_128.pth')
    network.eval()
    return network

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def euc_dis(a,b):
    return norm(a-b)

def identity_loss(model, img1, img2):
    img1_emb = return_embedding(model, img1)
    img2_emb = return_embedding(model, img2)
    return 1 - cos_sim(img1_emb, img2_emb)

def return_embedding(model, image):
    face_feats = np.empty((1,128)) # Embedding size
    im_array = np.array(image).reshape((1,3,224,224))
    f = model(torch.Tensor(im_array))[1].detach().cpu().numpy()[:, :, 0, 0]
    face_feats = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True)) # Normalizing embedding
    return face_feats.reshape(-1)

def return_activations(model, image):
    activations = list()
    normalized_values = 0
    im_array = np.array(image).reshape((1,3,224,224))
    for i in range(2,7):
        f = model(torch.Tensor(im_array))[i].detach().cpu().numpy()[:, :, 0, 0]
        normalized_values = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
        activations.append(normalized_values.reshape(-1))
    return np.array(activations)

def face_content_loss(model, img1, img2, metric='L2'):
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
    return lossclass FacePerceptionLoss(nn.Module):
    def __init__(self):
        super(FacePerceptualLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.model = initialize_senet50()

    def forward(self, out_images, target_images):
        
        # Identity Loss
        identity_loss = identity_loss(self.model, out_images, target_images)
        # Content Loss
        content_loss = face_content_loss(self.model, out_images, target_images)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss +  0.1 * identity_loss + 0.1 * content_loss

class FacePerceptionLoss(nn.Module):
    def __init__(self):
        super(FacePerceptualLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.model = initialize_senet50()

    def forward(self, out_images, target_images):
        
        # Identity Loss
        identity_loss = identity_loss(self.model, out_images, target_images)
        # Content Loss
        content_loss = face_content_loss(self.model, out_images, target_images)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        return image_loss +  0.1 * identity_loss + 0.1 * content_loss

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
