import time
from collections import OrderedDict
from options.test_options import TestOptions
from dataset.data_loader import CreateDataLoader
from dataset.aligned_dataset import my_Dataset
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
from random import randint
writer = SummaryWriter('runs/G1G2')
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


#opt = TestOptions().parse()
# iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
#
#model = create_model(opt)






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
# iter_start_time = time.time()
# save_fake = True
#
# person_folder = os.path.join('./MyWebApp/static/database', 'person')
# person_names = os.listdir(person_folder)
# idx = randint(0, len(person_names) - 1)
# person_path = os.path.join(person_folder, person_names[idx])
#
# cloth_folder = os.path.join('./MyWebApp/static/database', 'cloth')
# cloth_names = os.listdir(cloth_folder)
# cloth_path = os.path.join(cloth_folder, cloth_names[idx])

# person_path = './data/test2/person/0000_0.jpg'
# cloth_path = './data/test2/cloth/0000_1.jpg'
def build_model():
    opt = TestOptions().parse()
    model = create_model(opt)
    return opt, model
def try_on_database(opt, model, person_path, cloth_path):
    dataset = my_Dataset(opt, person_path, cloth_path)
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
#        a = generate_label_color(generate_label_plain(input_label)).float()
#        b = real_image.float()
        c = fake_image.float()

    img = (c[0].squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1)/2
    rgb = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr
    # d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
    # combine = torch.cat([a[0],d[0],b[0],c[0],rgb[0]], 2).squeeze()
    # #combine = c[0].squeeze()
    # # combine=c[0].squeeze()
    # cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    # # writer.add_image('combine', (combine.data + 1) / 2.0, step)
    # rgb=(cv_img*255).astype(np.uint8)
    # bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./sample/'+data['name'][0],bgr)
    #

# person_path = './data/test2/person/0000_0.jpg'
# cloth_path = './data/test2/cloth/0000_1.jpg'
# result = try_on_database(opt, model, person_path, cloth_path)
# cv2.imwrite('./sample/' + '1.jpg', result)



### display output images
    # if torch.cuda.is_available():
    #     a = generate_label_color(generate_label_plain(input_label)).float().cuda()
    #     b = real_image.float().cuda()
    #     c = fake_image.float().cuda()
    # else:
    #     a = generate_label_color(generate_label_plain(input_label)).float()
    #     b = real_image.float()
    #     c = fake_image.float()
    # d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
    # combine = torch.cat([a[0],d[0],b[0],c[0],rgb[0]], 2).squeeze()
    # #combine = c[0].squeeze()
    # # combine=c[0].squeeze()
    # cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    # # writer.add_image('combine', (combine.data + 1) / 2.0, step)
    # rgb=(cv_img*255).astype(np.uint8)
    # bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./sample/'+data['name'][0],bgr)
    #


############################ end test block ########################



# def changearm(old_label):
#     label=old_label
#     arm1=torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.int))
#     arm2=torch.FloatTensor((data['label'].cpu().numpy()==13).astype(np.int))
#     noise=torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.int))
#     label=label*(1-arm1)+arm1*4
#     label=label*(1-arm2)+arm2*4
#     label=label*(1-noise)+noise*4
#     return label
#
# opt.phase = 'valid'
# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# dataset_size = len(data_loader)
# print('# Inference images = %d' % dataset_size)
#
#
# step = 0
# for i, data in enumerate(dataset):
#     print(i)
#     iter_start_time = time.time()
#     # total_steps += opt.batchSize
#     # epoch_iter += opt.batchSize
#
#     # whether to collect output images
#     # save_fake = total_steps % opt.display_freq == display_delta
#     save_fake = True
#
#     ##add gaussian noise channel
#     ## wash the label
#     t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
#     print(data['label'].size())
#     #
#     # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
#     mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
#     mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
#     img_fore = data['image'] * mask_fore
#     img_fore_wc = img_fore * mask_fore
#     all_clothes_label = changearm(data['label'])
#
#     ############## Forward Pass ######################
#     if torch.cuda.is_available():
#         losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
#             Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()),
#             Variable(mask_clothes.cuda())
#             , Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()),
#             Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))
#     else:
#         losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
#             Variable(data['label']), Variable(data['edge']), Variable(img_fore),
#             Variable(mask_clothes)
#             , Variable(data['color']), Variable(all_clothes_label), Variable(data['image']),
#             Variable(data['pose']), Variable(data['image']), Variable(mask_fore))
#
#
#     ############## Display results and errors ##########
#
#     ### display output images
#     if torch.cuda.is_available():
#         a = generate_label_color(generate_label_plain(input_label)).float().cuda()
#         b = real_image.float().cuda()
#         c = fake_image.float().cuda()
#     else:
#         a = generate_label_color(generate_label_plain(input_label)).float()
#         b = real_image.float()
#         c = fake_image.float()
#     d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
#     combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze()
#     # combine = c[0].squeeze()
#     # combine=c[0].squeeze()
#     cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
#     cv2.imshow('image', cv_img)
#     if step % 1 == 0:
#         writer.add_image('combine', (combine.data + 1) / 2.0, step)
#         rgb = (cv_img * 255).astype(np.uint8)
#         bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#         n = str(step) + '.jpg'
#         cv2.imwrite('./sample/' + data['name'][0], bgr)
#     step += 1
#     print(step)
#
#
#     # ### save latest model
#     # if total_steps % opt.save_latest_freq == save_delta:
#     #     # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
#     #     # model.module.save('latest')
#     #     # np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
#     #     pass
#     # if epoch_iter >= dataset_size:
#     #     break
#     #
#     # # end of epoch
#     # iter_end_time = time.time()
#     # print('End of epoch %d / %d \t Time Taken: %d sec' %
#     #       (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
#     # break
#
#     # ### save model for this epoch
#     # if epoch % opt.save_epoch_freq == 0:
#     #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
#     #     model.module.save('latest')
#     #     model.module.save(epoch)
#     #     # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
#     #
#     # ### instead of only training the local enhancer, train the entire network after certain iterations
#     # if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
#     #     model.module.update_fixed_params()
#     #
#     # ### linearly decay learning rate after certain iterations
#     # if epoch > opt.niter:
#     #     model.module.update_learning_rate()
