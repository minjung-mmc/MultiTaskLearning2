# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:07:35 2020
@author: Santanu
"""

import numpy as np
import torch
from config import params

# out_seg = [1,128,256], seg = [1,128,256] (0~33)
class ConfusionMatrix:
    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual
        self.h = pred.size(1)
        self.w = pred.size(2)
        self.class_num = params.num_classes_seg

    def construct(self):

        assert self.pred.shape == self.actual.shape
        assert torch.min(self.pred) >= 0 and torch.min(self.actual >= 0)
        assert torch.max(self.pred) < 34 and torch.max(self.actual < 34)

        # print(self.pred.shape)
        # -------------converting into 1d array and then finding the frequency of each class-------------
        self.pred = self.pred.reshape(self.h * self.w)
        # storing the frequency of each class present in the predicted mask
        self.pred_count = torch.bincount(self.pred, weights=None, minlength=34)  # A
        self.actual = self.actual.reshape(self.h * self.w).int()
        # storing the frequency of each class present in the actual mask
        self.actual_count = torch.bincount(self.actual, weights=None, minlength=34)  # B
        print(self.pred_count)
        print(self.actual_count)
        # -----------------------------------------------------------------------------------------------

        """there are 21 classes but altogether 21x21=441 possibilities for every pixel
        for example, a pixel may actually belong to class '4' but may be predicted to be in class '3'
        So every pixel will have two features, one of which is actual and the other predicted
        To store both the details, we assign the category to which it belong
        Like in the above mentioned example the pixel belong to category 4-3"""

        # store the category of every pixel
        temp = self.actual * self.class_num + self.pred

        # frequency count of temp gives the confusion matrix 'cm' in 1d array format
        self.cm = torch.bincount(
            temp, weights=None, minlength=(self.class_num * self.class_num)
        )
        # reshaping the confusion matrix from 1d array to (no.of classes X no. of classes)
        self.cm = self.cm.reshape((self.class_num, self.class_num))

        # the diagonal values of cm correspond to those pixels which belong to same class in both predicted and actual mask
        self.Nr = torch.diag(self.cm, 0)  # A ⋂ B
        np.set_printoptions(threshold = np.inf, linewidth = np.inf)
        print("This is cm", self.cm)
        self.Dr = self.pred_count + self.actual_count - self.Nr  # A ⋃ B

    def computeMiou(self):
        individual_iou = self.Nr / self.Dr  # (A ⋂ B)/(A ⋃ B)
        print(individual_iou)
        sum = 0
        cnt = 0
        for i in range(len(individual_iou)):
            if (~torch.isnan(individual_iou[i])) and (individual_iou[i] != 0):
                sum += individual_iou[i]
                cnt += 1
        return sum / cnt

        # print("individual iou", individual_iou)
        # is_nan = torch.isnan(individual_iou)
        # individual_iou[is_nan] = 0
        # miou = (
        #     individual_iou.sum() / (~is_nan).float().sum()
        # )  # nanmean is used to neglect 0/0 case which arise due to absence of any class
        # return miou
