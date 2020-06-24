import sys
sys.path.append("./models/SingleHumanParser")
sys.path.append("/usr/local/python")
sys.path.append("/home/ubuntu/Insight_Project/openpose/python")
import shutil
import time
from collections import OrderedDict
from options.test_options import TestOptions
from dataset.data_loader import CreateDataLoader
from dataset.aligned_dataset import my_Dataset
from models.models import create_model
import util.util as util
from PIL import Image, ImageDraw
import os
import numpy as np
import torch
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
import cv2
from dataset.base_dataset import get_params, get_transform, normalize
from random import randint
from models.SingleHumanParser.inference1 import get_parser
import cv2 as cv
import json
#from openpose import pyopenpose as op

#writer = SummaryWriter('runs/G1G2')


SIZE=320
NC=14
def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)

    return label_batch

def generate_label_color(opt, inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)
    return input_label

def complete_compose(img,mask,label):
    label=label.cpu().numpy()
    M_f=label>0
    M_f=M_f.astype(np.int)
    M_f=torch.FloatTensor(M_f)
    masked_img=img*(1-mask)
    M_c=(1-mask.cuda())*M_f
    M_c=M_c+torch.zeros(img.shape)##broadcasting
    return masked_img,M_c,M_f

def compose(label,mask,color_mask,edge,color,noise):
    # check=check>0
    # print(check)
    masked_label=label*(1-mask)
    masked_edge=mask*edge
    masked_color_strokes=mask*(1-color_mask)*color
    masked_noise=mask*noise
    return masked_label,masked_edge,masked_color_strokes,masked_noise


########## Test Block ###############
def changearm(data):
    label=data['parser']
    arm1=torch.FloatTensor((data['parser'].cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((data['parser'].cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((data['parser'].cpu().numpy()==7).astype(np.int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label


def build_model():
    opt = TestOptions().parse()
    model = create_model(opt)
    return opt, model

#Get the mask of the cloth and put it in the desired destination
def get_item_mask(img):
  img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
  mask = img[:,:,0]
  ret,img_th = cv.threshold(mask,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
  # Copy the thresholded image.
  img_floodfill = img_th.copy()

  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = img_floodfill.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  # Floodfill from point (0, 0)
  cv.floodFill(img_floodfill, mask, (0,0), 255);
  # Invert floodfilled image
  img_floodfill_inv = cv.bitwise_not(img_floodfill)
  # Combine the two images to get the foreground.
  img_out = img_th | img_floodfill_inv
  return img_out


def try_on(opt, name, model, person_path, cloth_path, pose_path, from_user):
    if not from_user:
        print(from_user)
        parser_path = person_path.replace('.jpg', '.png').replace('person', 'person_parser')
        cloth_mask_path = cloth_path.replace('cloth', 'cloth_mask')
        dataset = my_Dataset(opt, name, person_path, cloth_path, pose_path, parser_path, cloth_mask_path, from_user)
    else:
        parser_path = person_path.replace('.jpg', '.png').replace('person', 'person_parser')
        parser_path = get_parser(person_path, parser_path)
        dataset = my_Dataset(opt, name, person_path, cloth_path, pose_path, parser_path)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
    for i, d in enumerate(data_loader):
        data = d
        break

    mask_clothes = torch.FloatTensor((data['parser'].cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor((data['parser'].cpu().numpy() > 0).astype(np.int))
    img_fore = data['person'] * mask_fore
    img_fore_wc = img_fore * mask_fore
    all_clothes_label = changearm(data)



    ############## Forward Pass ######################
    if torch.cuda.is_available():
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['parser'].cuda()), Variable(data['cloth_mask'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['cloth'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['person'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['person'].cuda()), Variable(mask_fore.cuda()))
    else:
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['parser']), Variable(data['cloth_mask']), Variable(img_fore),
            Variable(mask_clothes)
            , Variable(data['cloth']), Variable(all_clothes_label), Variable(data['person']),
            Variable(data['pose']), Variable(data['person']), Variable(mask_fore))

    #
    # ############## return as image ##########

    if torch.cuda.is_available():
#        a = generate_label_color(generate_label_plain(input_label)).float().cuda()
#        b = real_image.float().cuda()
        c = fake_image.float().cuda()
    else:
        c = fake_image.float()

    img = (c[0].squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1)/2
    rgb = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr






def try_on_test(opt, name, model, person_path, cloth_path, pose_path, from_user):
    if not from_user:
        print(from_user)
        parser_path = person_path.replace('.jpg', '.png').replace('person', 'person_parser')
        cloth_mask_path = cloth_path.replace('cloth', 'cloth_mask')
        dataset = my_Dataset(opt, name, person_path, cloth_path, pose_path, parser_path, cloth_mask_path, from_user)
    else:
        parser_path = person_path.replace('.jpg', '.png').replace('person', 'person_parser')
        parser_path = get_parser(person_path, parser_path)
        dataset = my_Dataset(opt, name, person_path, cloth_path, pose_path, parser_path)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
    for i, d in enumerate(data_loader):
        data = d
        break

    mask_clothes = torch.FloatTensor((data['parser'].cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor((data['parser'].cpu().numpy() > 0).astype(np.int))
    img_fore = data['person'] * mask_fore
    img_fore_wc = img_fore * mask_fore
    all_clothes_label = changearm(data)



    ############## Forward Pass ######################
    if torch.cuda.is_available():
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['parser'].cuda()), Variable(data['cloth_mask'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['cloth'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['person'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['person'].cuda()), Variable(mask_fore.cuda()))
    else:
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['parser']), Variable(data['cloth_mask']), Variable(img_fore),
            Variable(mask_clothes)
            , Variable(data['cloth']), Variable(all_clothes_label), Variable(data['person']),
            Variable(data['pose']), Variable(data['person']), Variable(mask_fore))

    #
    # ############## return as image ##########

    if torch.cuda.is_available():
        a = generate_label_color(opt, generate_label_plain(input_label)).float().cuda()
        b = real_image.float().cuda()
        c = fake_image.float().cuda()
        d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
        combine = torch.cat([a[0],d[0],b[0],c[0],rgb[0]], 2).squeeze()
    else:
        c = fake_image.float()

    img = (combine.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1)/2
    rgb = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr








#get_pose(person_path)
#shutil.copyfile(person_path, '../openpose/AppIn/1.jpg')
#os.chdir("../openpose")
#os.system("./build/examples/openpose/openpose.bin --image_dir AppIn --write_json AppOut/ --display 0 --render_pose 0 ")
#os.chdir("../Virtural_TryOn")
#bgr = try_on_input(opt, model, person_path, cloth_path, name)

if __name__ == "__main__":

    person_path = './data/valid/valid_img/000001_0.jpg'
    name = '000001_0.jpg'
    cloth_path = './data/valid/valid_color/000010_1.jpg'

    opt = TestOptions().parse()
    model = create_model(opt)
    person_img = Image.open(person_path)
    parser_path = person_path.replace('.jpg', '.png').replace('person', 'person_parser')
    parser_path = get_parser(person_path, parser_path)
    
    #model = None
    pose_path = "../openpose/examples/AppOut/1_keypoints.json"
    result = try_on_test(opt, name, model, person_path, cloth_path, pose_path, from_user = True)
    cv2.imwrite('./sample/result.jpg', result)
#    bgr = try_on_input(opt, model, person_path, cloth_path, pose_path, name)


    
