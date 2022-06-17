from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torchvision.models
import numpy as np
import math
import modules
import torch.utils.model_zoo as model_zoo

# class gaze_model(nn.Module):
#     def __init__(self):
#         super(gaze_model, self).__init__()
#         self.feature =  modules.resnet152(pretrained=True)
#         # self.feature.load_state_dict(torch.load(pretrained_url), strict=False )

#         self.gazeEs = modules.ResGazeEs()
#         # self.gazeEs.load_state_dict(torch.load(pretrained_url), strict=False )

#         # self.deconv = modules.ResDeconv(modules.BasicBlock)

#     def forward(self, x_in, require_img=True):
#         features = self.feature(x_in['face'])
#         gaze = self.gazeEs(features)
#         # if require_img:
#         #   img = self.deconv(features)
#         #   img = torch.sigmoid(img)
#         # else:
#         #   img = None
#         # return gaze, img
#         return gaze

class gaze_model(nn.Module):
    def __init__(self):
        super(gaze_model, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        
        vgg16ForLeft = torchvision.models.vgg16(pretrained=True)
        vgg16ForRight = torchvision.models.vgg16(pretrained=True)

        self.leftEyeNet = vgg16ForLeft.features
        self.rightEyeNet = vgg16ForRight.features

        self.leftPool = nn.AdaptiveAvgPool2d(1)
        self.rightPool = nn.AdaptiveAvgPool2d(1)

        self.leftFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.rightFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC2 = nn.Sequential(
            nn.Linear(514, 256),
            nn.BatchNorm1d(256, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        # self._init_weights()

    def forward(self, x_in):
        leftFeature = self.leftEyeNet(x_in['left'])
        leftFeature = self.leftPool(leftFeature)
        leftFeature = leftFeature.view(leftFeature.size(0), -1)
        leftFeature = self.leftFC(leftFeature)

        rightFeature = self.rightEyeNet(x_in['right'])
        rightFeature = self.rightPool(rightFeature)
        rightFeature = rightFeature.view(rightFeature.size(0), -1)
        rightFeature = self.rightFC(rightFeature)

        feature = torch.cat((leftFeature, rightFeature), 1)

        feature = self.totalFC1(feature)
        feature = torch.cat((feature,  x_in['head_pose']), 1)

        gaze = self.totalFC2(feature)

        return gaze
    
class Gelossop():
    def __init__(self, attentionmap, w1=1, w2=1):
        self.gloss = torch.nn.L1Loss().cuda()
        #self.gloss = torch.nn.MSELoss().cuda()
        self.recloss = torch.nn.MSELoss().cuda()
        self.attentionmap = attentionmap.cuda()
        self.w1 = w1
        self.w2 = w2
        

    def __call__(self, gaze, img, gaze_pre, img_pre):
        loss1 = self.gloss(gaze, gaze_pre)
        # loss2 = 1-self.recloss(img, img_pre)
        loss2 = 1 - (img - img_pre)**2
        zeros = torch.zeros_like(loss2)
        loss2 = torch.where(loss2 < 0.75, zeros, loss2)
        loss2 = torch.mean(self.attentionmap * loss2)

        return self.w1 * loss1 + self.w2 * loss2

class Delossop():
    def __init__(self):
        self.recloss = torch.nn.MSELoss().cuda()

    def __call__(self, img, img_pre):
        return self.recloss(img, img_pre)