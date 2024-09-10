''' Pytorch Recursive CNN Trainer
 Authors : Khurram Javed
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function
from typing import Tuple, List

import logging

from torch.autograd import Variable

logger = logging.getLogger('iCARL')
import torch.nn.functional as F
import torch
import wandb
import numpy as np
import pandas as pd
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

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        for img, target,_ in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            response = self.model(Variable(img))
            # print (response[0])
            # print (target[0])
            loss = F.mse_loss(response, Variable(target.float()))
            loss = torch.sqrt(loss)
            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})



class Trainer_with_class(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer,loss="mse"):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        if loss=="mse":
            self.loss_funct=F.mse_loss
        elif loss=="l1":
            self.loss_funct=F.l1_loss

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        classification_results=[]

        for img, target,_ in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            model_prediction = self.model(Variable(img))
            x_cords = model_prediction[:, [0, 2, 4, 6]]
            y_cords = model_prediction[:, [1, 3, 5, 7]]
            classification_result = self.evaluate_corners(x_cords, y_cords, target, _)
            classification_results.extend(classification_result)
            loss = self.loss_funct(model_prediction, Variable(target.float()))
            loss = torch.sqrt(loss)
            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        # wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})
        df = pd.DataFrame(classification_results)
        # lossAvg /= len(iterator)
        total_corners = df["total_corners"]
        wandb.log({"epoch": epoch,
                   "avg_train_loss": lossAvg,
                   "train_accuracy": (np.sum(total_corners)/4)/len(total_corners),
                   "train_4_corners_accuracy": np.sum(total_corners == 4) / len(total_corners),
                   "train_3_corners_accuracy": np.sum(total_corners >= 3) / len(total_corners),

                   })
    def cordinate_within_intervals(self, cordinate, x_interval, y_interval) -> int:

        is_within_x = (x_interval[0] <= cordinate[0] <= x_interval[1])
        is_within_y = (y_interval[0] <= cordinate[1] <= y_interval[1])

        return int(is_within_x and is_within_y)


    def evaluate_corners(self, x_cords: np.ndarray, y_cords: np.ndarray, target: np.ndarray,paths:str) -> Tuple[List,List]:

        target = target.cpu().data.numpy()
        target_x = target[:, [0, 2, 4, 6]]
        target_y = target[:, [1, 3, 5, 7]]
        resut_dicts=[]

        for entry in range(len(target)):
            top_left_y_lower_bound = max(0, (2 * y_cords[entry, 0] - (y_cords[entry, 3] + y_cords[entry, 0]) / 2))
            top_left_y_upper_bound = ((y_cords[entry, 3] + y_cords[entry, 0]) / 2)
            top_left_x_lower_bound = max(0, (2 * x_cords[entry, 0] - (x_cords[entry, 1] + x_cords[entry, 0]) / 2))
            top_left_x_upper_bound = ((x_cords[entry, 1] + x_cords[entry, 0]) / 2)

            top_right_y_lower_bound = max(0, (2 * y_cords[entry, 1] - (y_cords[entry, 1] + y_cords[entry, 2]) / 2))
            top_right_y_upper_bound = ((y_cords[entry, 1] + y_cords[entry, 2]) / 2)
            top_right_x_lower_bound = ((x_cords[entry, 1] + x_cords[entry, 0]) / 2)
            top_right_x_upper_bound = min(1, (x_cords[entry, 1] + (x_cords[entry, 1] - x_cords[entry, 0]) / 2))

            bottom_right_y_lower_bound = ((y_cords[entry, 1] + y_cords[entry, 2]) / 2)
            bottom_right_y_upper_bound = min(1, (y_cords[entry, 2] + (y_cords[entry, 2] - y_cords[entry, 1]) / 2))
            bottom_right_x_lower_bound = ((x_cords[entry, 2] + x_cords[entry, 3]) / 2)
            bottom_right_x_upper_bound = min(1, (x_cords[entry, 2] + (x_cords[entry, 2] - x_cords[entry, 3]) / 2))

            bottom_left_y_lower_bound = ((y_cords[entry, 0] + y_cords[entry, 3]) / 2)
            bottom_left_y_upper_bound = min(1, (y_cords[entry, 3] + (y_cords[entry, 3] - y_cords[entry, 0]) / 2))
            bottom_left_x_lower_bound = max(0, (2 * x_cords[entry, 3] - (x_cords[entry, 2] + x_cords[entry, 3]) / 2))
            bottom_left_x_upper_bound = ((x_cords[entry, 3] + x_cords[entry, 2]) / 2)

            top_left = (target_x[entry,0],target_y[entry,0])
            top_right = (target_x[entry,1],target_y[entry,1])
            bottom_right = (target_x[entry,2],target_y[entry,2])
            bottom_left = (target_x[entry,3],target_y[entry,3])

            tl = self.cordinate_within_intervals(top_left, (top_left_x_lower_bound, top_left_x_upper_bound),
                                                 (top_left_y_lower_bound,
                                                  top_left_y_upper_bound))
            tr = self.cordinate_within_intervals(top_right, (top_right_x_lower_bound, top_right_x_upper_bound),
                                                 (top_right_y_lower_bound,
                                                  top_right_y_upper_bound))
            br = self.cordinate_within_intervals(bottom_right, (bottom_right_x_lower_bound, bottom_right_x_upper_bound),
                                                 (bottom_right_y_lower_bound,
                                                  bottom_right_y_upper_bound))
            bl = self.cordinate_within_intervals(bottom_left, (bottom_left_x_lower_bound, bottom_left_x_upper_bound),
                                                 (bottom_left_y_lower_bound,
                                                  bottom_left_y_upper_bound))
            resut_dict = {"path":paths[entry],
                "contains_tl": tl,
                           "contains_tr": tr,
                           "contains_br": br,
                           "contains_bl": bl,
                           "total_corners": tl + tr + br + bl,

                "top_left": (top_left,
                             top_left_y_lower_bound,
                             top_left_y_upper_bound,
                             top_left_x_lower_bound,
                             top_left_x_upper_bound),
                "top_right": (top_right,
                              top_right_y_lower_bound,
                              top_right_y_upper_bound,
                              top_right_x_lower_bound,
                              top_right_x_upper_bound),
                "bottom_right": (bottom_right,
                                 bottom_right_y_lower_bound,
                                 bottom_right_y_upper_bound,
                                 bottom_right_x_lower_bound,
                                 bottom_right_x_upper_bound),
                "bottom_left": (bottom_left,
                                bottom_left_y_lower_bound,
                                bottom_left_y_upper_bound,
                                bottom_left_x_lower_bound,
                                bottom_left_x_upper_bound),
            }
            resut_dicts.append(resut_dict)
        return resut_dicts





class CompleteDocumentTrainer(GenericTrainer):
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

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        loss_function = torch.nn.CrossEntropyLoss()
        for img, target ,type,dataset,path in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            response = self.model(Variable(img))

            loss =loss_function(response, Variable(target.float()))

            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss

            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})

class CIFARTrainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

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
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in tqdm(self.train_iterator):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(Variable(inputs), pretrain=True)
            loss = self.criterion(outputs, Variable(targets))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        logger.info("Accuracy : %s", str((correct * 100) / total))
        return correct / total
