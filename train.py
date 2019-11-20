import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils import batch

from matplotlib.image import imread
from skimage.transform import rescale

image_dir = '../Linlin/img'
groundtruth_dir = '../Linlin/groundTruth'
dir_checkpoint = 'checkpoints/'

def test_imgs_and_masks(scale=0.4):
    img_paths = sorted([os.path.join(image_dir, s) for s in os.listdir(image_dir) if 'hh' in s])

    for hh in img_paths:
        hv = hh.replace('-hh-', '-hv-')[:-8]+'002.tiff'
        hh_channel = imread(hh)
        hv_channel = imread(hv)
        
        hh_channel = rescale(hh_channel, scale, anti_aliasing=True)
        hv_channel = rescale(hv_channel, scale, anti_aliasing=True)
        image = np.stack((hh_channel, hv_channel), axis=0)

        mask = imread(hh.replace('-hh-', '-gt-test-').replace('img', 'groundTruth')[:-8]+'000.bmp')
        mask = rescale(mask, scale, order=0, anti_aliasing=False)*255 # nearest neighbour 
        mask = np.expand_dims(mask, axis=0)
        yield image, mask

def train_imgs_and_masks(scale=0.4):
    img_paths = sorted([os.path.join(image_dir, s) for s in os.listdir(image_dir) if 'hh' in s])[:-1]

    for hh in img_paths:
        hv = hh.replace('-hh-', '-hv-')[:-8]+'002.tiff'
        hh_channel = imread(hh)
        hv_channel = imread(hv)
        
        hh_channel = rescale(hh_channel, scale, anti_aliasing=True)
        hv_channel = rescale(hv_channel, scale, anti_aliasing=True)
        image = np.stack((hh_channel, hv_channel), axis=0)

        mask = imread(hh.replace('-hh-', '-gt-train-').replace('img', 'groundTruth')[:-8]+'000.bmp')
        mask = rescale(mask, scale, order=0, anti_aliasing=False)*255 # nearest neighbour 
        mask = np.expand_dims(mask, axis=0)
        yield image, mask

def val_imgs_and_masks(scale=0.4):
    img_paths = sorted([os.path.join(image_dir, s) for s in os.listdir(image_dir) if 'hh' in s])[-1:]

    for hh in img_paths:
        hv = hh.replace('-hh-', '-hv-')[:-8]+'002.tiff'
        hh_channel = imread(hh)
        hv_channel = imread(hv)
        
        hh_channel = rescale(hh_channel, scale, anti_aliasing=True)
        hv_channel = rescale(hv_channel, scale, anti_aliasing=True)
        image = np.stack((hh_channel, hv_channel), axis=0)

        mask = imread(hh.replace('-hh-', '-gt-train-').replace('img', 'groundTruth')[:-8]+'000.bmp')
        mask = rescale(mask, scale, order=0, anti_aliasing=False)*255 # nearest neighbour 
        mask = np.expand_dims(mask, axis=0)
        yield image, mask

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              img_scale=0.4,
              save_cp=True,
              startepoch=0):
    
    

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    n_train = 18
    n_val = 1

    val = val_imgs_and_masks(img_scale)

    val_score = eval_net(net, val, device, n_val)
    if net.n_classes > 1:
        logging.info('Validation cross entropy: {}'.format(val_score))

    else:
        logging.info('Validation Dice Coeff: {}'.format(val_score))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        # reset the generators
        train = train_imgs_and_masks(img_scale)
        val = val_imgs_and_masks(img_scale)

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + startepoch + 1}/{epochs + startepoch}', unit='img') as pbar:
            for b in batch(train, batch_size):
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b]).astype(np.float32)

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)

                imgs = imgs.to(device=device)
                true_masks = true_masks.to(device=device)

                masks_pred = net(imgs)

                masks_pred = masks_pred[true_masks!=255]
                true_masks = true_masks[true_masks!=255]

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)

            pbar.set_postfix(**{'loss (epoch)': epoch_loss})

        if save_cp and ((epoch+1)%1 == 0 or (epoch+1)==epochs):
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + startepoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + startepoch + 1} saved !')

        if (epoch+1)%50 == 0:
            val_score = eval_net(net, val, device, n_val)
            if net.n_classes > 1:
                logging.info('Validation cross entropy: {}'.format(val_score))

            else:
                logging.info('Validation Dice Coeff: {}'.format(val_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-se', '--start-epoch', dest='startepoch', type=int, default=False,
                        help='Load model from checkpoints/CP_epoch(input).pth')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.4,
                        help='Downscaling factor of the images')

    return parser.parse_args()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=2, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    elif args.startepoch:
        load = os.path.join(dir_checkpoint, 'CP_epoch{}.pth'.format(args.startepoch))
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
        logging.info(f'Model loaded from {load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                device=device,
                img_scale=args.scale,
                startepoch=args.startepoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
