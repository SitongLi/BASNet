# Modified for new version torch
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


bce_loss = nn.BCELoss(size_average=True)
focal_loss = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred, target):

    bce_out = bce_loss(pred, target)
    focal_out = focal_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = focal_out + ssim_out + iou_out
    loss_old = bce_out + ssim_out + iou_out

    return loss_old, loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

    old0, loss0 = bce_ssim_loss(d0, labels_v)
    old1, loss1 = bce_ssim_loss(d1, labels_v)
    old2, loss2 = bce_ssim_loss(d2, labels_v)
    old3, loss3 = bce_ssim_loss(d3, labels_v)
    old4, loss4 = bce_ssim_loss(d4, labels_v)
    old5, loss5 = bce_ssim_loss(d5, labels_v)
    old6, loss6 = bce_ssim_loss(d6, labels_v)
    old7, loss7 = bce_ssim_loss(d7, labels_v)
    #ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + \
        loss4 + loss5 + loss6 + loss7  # + 5.0*lossa
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, l7: %3f" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item(), loss7.data.item()))
    print("loss_old l0:%3f, l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, l6:%3f, l7:%3f\n" % (
        old0.data.item(), old1.data.item(), old2.data.item(), old3.data.item(), old4.data.item(), old5.data.item(), old6.data.item(), old7.data.item()))
    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = '../SOD_datasets/'
tra_image_dir = 'DUTS-TR/im_aug/'
tra_label_dir = 'DUTS-TR/gt_aug/'

image_ext = '.jpg'
label_ext = '.png'

model_dir = "./saved_models/basnet_bsi/"


epoch_num = 100000
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(
    salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1)
if torch.cuda.is_available():
    net.cuda()

net = nn.DataParallel(net, device_ids=[0,1])

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(
    0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % 2000 == 0:  # save model every 2000 iterations

            torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" %
                       (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
