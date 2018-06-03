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
from tqdm import tqdm

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="mse", cuda=True):
        if testType == "mse":
            return DocumentMseEvaluator(cuda)



class DocumentMseEvaluator():
    '''
    Evaluator class for softmax classification 
    '''
    def __init__(self, cuda):
        self.cuda = cuda


    def evaluate(self, model, iterator):
        model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))
                # print (response[0])
                # print (target[0])
                loss = F.l1_loss(response, Variable(target.float()))
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(iterator)
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))


