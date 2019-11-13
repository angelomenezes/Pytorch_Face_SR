from math import log10
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.data_loader_YCbCr import *
from SUB_CNN_model import Net
from utils.pytorch_ssim import *

torch.manual_seed(1)
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='PyTorch SubPixelCNN')

# hyper-parameters
parser.add_argument('--inputDir', type=str, help='where the low-res data belong')
parser.add_argument('--targetDir', type=str, help='where the target data belong')
parser.add_argument('--upscale_factor', type=int, default=4, help="super-resolution upscale factor")

args = parser.parse_args()

def main():
    
    # SubPixelCNN parameters

    batch_size = 32
    epochs = 50
    lr = 0.01
    threads = 4
    upscale_factor = args.upscale_factor

    img_path_low = args.inputDir
    img_path_ref = args.targetDir

    train_set = DatasetSuperRes(img_path_low, img_path_ref)

    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, 
                                      batch_size=batch_size, shuffle=True)

    model = Net(upscale_factor=upscale_factor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    out_path = 'results/'
    out_model_path = 'models/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)    

    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)   
        
    results = {'avg_loss': [], 'psnr': [], 'ssim': []}
    
    # Training

    begin_counter = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            input_, target = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            upsampled_img = model(input_)
            loss = criterion(upsampled_img, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
        
        scheduler.step() # Decrease learning rate after 20 epochs to 20% of its value
        
        psnr_epoch = 10*log10(1/(epoch_loss / len(training_data_loader)))
        ssim_epoch = ssim(upsampled_img, target).item()
        avg_loss_batch = epoch_loss/len(training_data_loader)
        
        results['psnr'].append(psnr_epoch)
        results['ssim'].append(ssim_epoch)
        results['avg_loss'].append(avg_loss_batch)
        
        print("===> Epoch {} Complete: Avg. Loss: {:.4f} / PSNR: {:.4f} / SSIM {:.4f}".format(epoch, 
                                                                                            avg_loss_batch, 
                                                                                            psnr_epoch,
                                                                                            ssim_epoch))
        # Checkpoint
        if epoch % (epochs // 2) == 0:
        
            data_frame = pd.DataFrame(
                    data={'Avg. Loss': results['avg_loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))

            data_frame.to_csv(out_path + 'SubCNN_x' + str(upscale_factor) + '_train_results.csv', index_label='Epoch')            
            path = out_model_path + "SubCNN_x{}_epoch_{}.pth".format(upscale_factor, epoch)
            torch.save(model, path)
            print("Checkpoint saved to {}".format(path))

    end_counter = time.time()
    training_time = end_counter - begin_counter
    print("Seconds spent during training = ", training_time)
    report = open(out_path + "SubCNN_model_x" + str(args.upscale_factor) + ".txt", "w")
    report.write("Training time: {:.2f}".format(training_time))
    report.close()

if __name__ == "__main__":
    main()
