#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import os
# print(os.listdir())
# sys.path.append('/models/SingleHumanParser')
import os
import argparse
import logging
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

#from net.pspnet import PSPNet
from net.pspnet import PSPNet
import cv2 as cv
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
parser.add_argument('--image_path', type=str, default='', help='Path to image')
parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")
args = parser.parse_args()
args.models_path = './models/SingleHumanParser/checkpoints'
# image_folder = '/Users/qingyunw/Downloads/Origin_Data/train'
#
# image_names = sorted(os.listdir(image_folder))
# image_path = os.path.join(image_folder, image_names[1])
# parser.image_path = image_path
# img = Image.open(image_path)

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu')))
        #net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    if torch.cuda.is_available():
        net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        #transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)


def show_image(img, pred, name):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))

    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=0.7, )
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(2.3, (j + 0.45) / 20.0, lab, ha='left', va='center', )

    plt.savefig(fname="./"  + name)
    print('result saved to ...')
    plt.show()

def class_change(n):
    # classes = np.array(('Background',  # always index 0
    #                     'Hat', 'Hair', 'Glove', 'Sunglasses',
    #                     'UpperClothes', 'Dress', 'Coat', 'Socks',
    #                     'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
    #                     'Face', 'Left-arm', 'Right-arm', 'Left-leg',
    #                     'Right-leg', 'Left-shoe', 'Right-shoe',))
        # 0 -> Background
        # 1 -> Hair
        # 4 -> Upclothes
        # 5 -> Left-shoe 
        # 6 -> Right-shoe
        # 8 -> Pants
        # 9 -> Left_leg
        # 10 -> Right_leg
        # 11 -> Left_arm
        # 12 -> Face
        # 13 -> Right_arm
    table = {
      0:0,
      1:1,
      2:1,
      3:0,
      4:12,
      5:4,
      6:4,
      7:4,
      8:0,
      9:8,
      10:4,
      11:4,
      12:4,
      13:12,
      14:11,
      15:13,
      16:9,
      17:10,
      18:5,
      19:6
    }
    return table[n]

def main():
    # --------------- model --------------- #
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    # ------------ load image ------------ #
    data_transform = get_transform()
    img = Image.open(image_path)
    #img = Image.open(open(args.image_path, 'rb'))
    img = data_transform(img)
    if torch.cuda.is_available():
        img = img.cuda()

    # --------------- inference --------------- #

    with torch.no_grad():
        pred, _ = net(img.unsqueeze(dim=0))
        pred = pred.squeeze(dim=0)
        pred = pred.cpu().numpy().transpose(1, 2, 0)
        #pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 192, 1))
        vfunc = np.vectorize(class_change)
        pred_change = vfunc(pred)
        # status = cv.imwrite('/content/gdrive/My Drive/Insight_Project/DeepFashion_Try_On/Data_preprocessing/test_label/0000_0.png',pred_change)
        status = cv.imwrite('./sample/0000_1.png',pred_change)

        show_image(img, pred, '0_')
        show_image(img, pred_change, '1_')

def get_parser(img, name):
    # --------------- load model --------------- #
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    # ------------ load image ------------ #
    data_transform = get_transform()
    img = data_transform(img)
    if torch.cuda.is_available():
        img = img.cuda()

    # --------------- inference --------------- #

    with torch.no_grad():
        pred, _ = net(img.unsqueeze(dim=0))
        pred = pred.squeeze(dim=0)
        pred = pred.cpu().numpy().transpose(1, 2, 0)
        #pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 192, 1))
        vfunc = np.vectorize(class_change)
        pred_change = vfunc(pred)
        # status = cv.imwrite('/content/gdrive/My Drive/Insight_Project/DeepFashion_Try_On/Data_preprocessing/test_label/0000_0.png',pred_change)
        status = cv.imwrite('./sample/0000_1.png',pred_change)
        #
        # colormap = [(0, 0, 0),
        #             (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
        #             (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
        #             (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
        #             (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
        #             (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
        # cmap = matplotlib.colors.ListedColormap(colormap)
        # bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # h, w, _ = pred_change.shape
        #
        # pred_change = pred_change.reshape((h, w))
        #
        # # show image
        # plt.imshow(pred_change, cmap=cmap, norm=norm)
        # plt.savefig(fname="./" + name)
        # print('result saved to ...')
    return Image.open('./sample/0000_1.png')


if __name__ == '__main__':
    main()
