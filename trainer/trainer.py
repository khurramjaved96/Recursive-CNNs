''' Pytorch Recursive CNN Trainer
 Authors : Khurram Javed
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import logging
from torch.autograd import Variable
logger = logging.getLogger('iCARL')
import torch.nn.functional as F
import torch
from tqdm import tqdm
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self):
        pass


class Trainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch):
        self.model.train()
        lossAvg=None
        for img, target in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()

            response = self.model(Variable(img))
            # print (response[0])
            # print (target[0])
            loss = F.mse_loss(response, Variable(target.float()))
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg+= loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg/=len(self.train_iterator)
        logger.info("Avg Loss %s", str(torch.sqrt(lossAvg).cpu().data.numpy()))
