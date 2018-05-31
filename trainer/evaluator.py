''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="nmc", cuda=True):
        if testType == "trainedClassifier":
            return softmax_evaluator(cuda)



class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''
    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))

    def evaluate(self, model, loader, scale=None, thres=False, older_classes=None, step_size=10, descriptor=False,
                 falseDec=False, higher=False):
        pass

