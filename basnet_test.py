# Modified for new version torch
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

# --------- 1. get image path and name ---------


#image_dir = './test_data/test_images/'
image_dir = '../SOD_datasets/DUTS-TE/img/'
prediction_dir = './test_data/test_results/'
model_dir = './saved_models/basnet_bsi_original/basnet_bsi_itr_574000_train_1.013724_tar_0.054404.pth'

img_name_list = glob.glob(image_dir + '*.jpg')

# --------- 2. dataloader ---------
# 1. dataload
test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[
], transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(
    test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

# --------- 3. model define ---------
print("...load BASNet...")
net = BASNet(3, 1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
    net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

    print("inferencing:", img_name_list[i_test].split("/")[-1])

    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    # save results to test_results folder
    save_output(img_name_list[i_test], pred, prediction_dir)
    sup_dir = prediction_dir+'_sup'
    if not os.path.exists(sup_dir):
        os.mkdir(sup_dir)
    pred3 = d3[:0, :, :]
    pred3 = normPRED(pred3)
    save_output(img_name_list[i_test]+'_d3', pred3, sup_dir)

    pred3 = d7[:0, :, :]
    pred7 = normPRED(pred7)
    save_output(img_name_list[i_test]+'_d7', pred7, sup_dir)

    del d1, d2, d3, d4, d5, d6, d7, d8
