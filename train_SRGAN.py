from math import log10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from super_resolution_data_loader_GAN import *
from pytorch_ssim import *

from SRGAN_model import Generator, Discriminator
from loss import GeneratorLoss

torch.manual_seed(1)
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='PyTorch SRGAN')

# hyper-parameters
parser.add_argument('--inputDir', type=str, help='where the low-res data belong')
parser.add_argument('--targetDir', type=str, help='where the target data belong')
parser.add_argument('--upscale_factor', type=int, default=4, help="super-resolution upscale factor")

args = parser.parse_args()

def main():
    
    # SRGAN parameters

    batch_size = 10
    num_epochs = 100
    lr = 0.01
    threads = 4
    upscale_factor = args.upscale_factor

    img_path_low = args.inputDir
    img_path_ref = args.targetDir

    train_set = DatasetSuperRes(img_path_low, img_path_ref)

    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, 
                                      batch_size=batch_size, shuffle=True)

    netG = Generator(upscale_factor).to(device)
    print('# Generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator().to(device)
    print('# Discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion = GeneratorLoss().to(device)    
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    out_path = 'results/'
    out_model_path = 'models/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)    

    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)   
        
    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}
    
    # Training

    begin_counter = time.time()

    for epoch in range(1, num_epochs + 1):
        
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 
                           'd_score': 0, 'g_score': 0}
        netG.train()
        netD.train()

        for data, target in training_data_loader:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target).to(device)
            z = Variable(data).to(device)
            
            fake_img = netG(z)

            netD.zero_grad()
            
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            
            optimizerD.step()
            
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # Loss for current batch before optimization 

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
         
        print('[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}'.format(
            epoch, num_epochs, 
            running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()

        batch_mse = ((fake_img - real_img) ** 2).data.mean()
        batch_ssim = ssim(fake_img, real_img).item()
        batch_psnr = 10 * log10(1 /batch_mse)
        
        results['psnr'].append(psnr_epoch)
        results['ssim'].append(ssim_epoch)
        results['avg_loss'].append(avg_loss_batch)
        
        # Save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(batch_psnr)
        results['ssim'].append(batch_ssim)

        # Checkpoint
        if epoch % (epochs // 10) == 0:
            # Save model
            torch.save(netG, out_model_path + 'netG_x%d_epoch_%d.pth' % (upscale_factor, epoch))
            #torch.save(netD, 'netD_x%d_epoch_%d.pt' % (upscale_factor, epoch))

            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                    'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'SRGAN_x' + str(upscale_factor) + '_train_results.csv', index_label='Epoch')

    end_counter = time.time()
    training_time = end_counter - begin_counter
    print("Seconds spent during training = ", training_time)
    report = open(out_path + "SRGAN_model_x" + str(args.upscale_factor) + ".txt", "w")
    report.write("Training time: {:.2f}".format(training_time))
    report.close()

if __name__ == "__main__":
    main()
