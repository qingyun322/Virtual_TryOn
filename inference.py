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
        dataset = my_Dataset(opt, name, person_path, cloth_path, pose_path)
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


def try_on_input(opt, model, person_path, cloth_path, pose_path, name):

    shutil.copyfile(person_path, '../openpose/examples/AppIn/1.jpg')
    #preparing dataset
    person_img = Image.open(person_path)
    cloth_img = Image.open(cloth_path)
    parser = get_parser(person_img, name).convert('L')
    params = get_params(opt, parser.size)
    if opt.label_nc == 0:
        transform_A = get_transform(opt, params)
        parser_tensor = transform_A(parser.convert('RGB'))
    else:
        transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        parser_tensor = transform_A(parser) * 255.0

    person = person_img.convert('RGB')
    transform_B = get_transform(opt, params)
    person_tensor = transform_B(person)

    cloth = cloth_img.convert('RGB')
    cloth_tensor = transform_B(cloth)

    cloth_mask = get_item_mask(cv.imread(cloth_path))
    cloth_mask = Image.fromarray(cloth_mask).convert('L')
    cloth_mask_tensor = transform_A(cloth_mask)

    ##Pose
    
    #os.chdir("../openpose")
    #os.system("./build/examples/openpose/openpose.bin --image_dir examples/AppIn --write_json examples/AppOut/ --display 0 --render_pose 0 ")
    #os.chdir("../Virtural_TryOn")

    with open(os.path.join(pose_path), 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))

    fine_height = 256
    fine_width = 192
    r = 5

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, fine_height, fine_width)
    im_pose = Image.new('L', (fine_width, fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (fine_width, fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i, 0]
        pointy = pose_data[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
        one_map = transform_B(one_map.convert('RGB'))
        pose_map[i] = one_map[0]
    pose_tensor = pose_map

  #  name = person_path.split('/')[-1]
  #  input_dict = {'parser': parser_tensor, 'label_ref': parser_tensor, 'person': person_tensor,
  #                'cloth_mask': cloth_mask_tensor,
  #                'cloth': cloth_tensor, 'image_ref': person_tensor, 'path': parser_path, 'path_ref': parser_path,
  #                'pose': pose_tensor, 'name': name}


    data = {'parser':parser_tensor, 'person':person_tensor, 'cloth_mask':cloth_mask_tensor, 'cloth':cloth_tensor, 'pose':pose_tensor}
 #   dataset = my_Dataset(opt, person_path, cloth_path)
 #   data_loader = torch.utils.data.DataLoader(
 #           dataset,
 #           batch_size=1,
 #           shuffle=not opt.serial_batches,
 #           num_workers=int(opt.nThreads))
 #   for i, d in enumerate(data_loader):
 #       data = d
 #       break
    
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
#        a = generate_label_color(generate_label_plain(input_label)).float()
#        b = real_image.float()
        c = fake_image.float()

    img = (c[0].squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1)/2
    rgb = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr






#Set parameters for openpose
# Set
def set_op_params():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "COCO"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["image_dir"] = './data/valid/valid_img'
    # If GPU version is built, and multiple GPUs are available, set the ID here
    #params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["write_json"] = "./"
    params["model_folder"] = "/Users/qingyunw/OneDrive/Programming/Insight_Toronto_2020/Forked_Projects/OpenPose/models"
    return params

# get pose map using openpose
def get_pose(person_path):
    params = set_op_params()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()
    imageToProcess = cv2.imread(person_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    print("Body keypoints: \n" + str(datum.poseKeypoints))


#get_pose(person_path)
#shutil.copyfile(person_path, '../openpose/AppIn/1.jpg')
#os.chdir("../openpose")
#os.system("./build/examples/openpose/openpose.bin --image_dir AppIn --write_json AppOut/ --display 0 --render_pose 0 ")
#os.chdir("../Virtural_TryOn")
#bgr = try_on_input(opt, model, person_path, cloth_path, name)

if __name__ == "__main__":

    person_path = './data/valid/valid_img/000001_0.jpg'
    name = '000001_0.jpg'
    cloth_path = './data/valid/valid_color/000001_1.jpg'

    opt = TestOptions().parse()
    model = create_model(opt)
#    model = None
    pose_path = "../openpose/examples/AppOut/1_keypoints.json"
    bgr = try_on_input(opt, model, person_path, cloth_path, pose_path, name)


    
