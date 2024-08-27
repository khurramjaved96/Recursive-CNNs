''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging
from sys import prefix

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import wandb
from torchnet.meter import confusionmeter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="rmse", cuda=True):
        if testType == "rmse":
            return DocumentMseEvaluator(cuda)
        if testType == "cross_entropy":
            return CompleteDocEvaluator(cuda)



class DocumentMseEvaluator():
    '''
    Evaluator class for softmax classification 
    '''
    def __init__(self, cuda):
        self.cuda = cuda


    def evaluate(self, model, iterator,epoch):
        model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))
                # print (response[0])
                # print (target[0])
                loss = F.mse_loss(response, Variable(target.float()))
                loss = torch.sqrt(loss)
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(iterator)
        wandb.log({"epoch":epoch,"eval_loss": lossAvg})
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))


import torch
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class CompleteDocEvaluator():
    '''
    Evaluator class for softmax classification
    '''

    def __init__(self, cuda):
        self.cuda = cuda

    def evaluate(self, model, iterator, epoch, prefix,table=True):
        model.eval()

        test_table=wandb.Table(columns=["img","directory","prediction","label","dataset","path"])

        lossAvg = None
        all_targets = []
        all_predictions = []
        loss_function = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for img, target,directory,dataset,path in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))

                loss = loss_function(response, Variable(target.float()))

                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss

                predictions = torch.argmax(response, dim=1).cpu().numpy()
                target = torch.argmax(target, dim=1).cpu().numpy()
                all_targets.extend(target)
                all_predictions.extend(predictions)
                img=np.transpose(img.cpu().numpy(),(0,2, 3, 1))
                if table:
                    for idx in range(len(target)):
                        numpy_img=wandb.Image(img[idx])
                        test_table.add_data(numpy_img,directory[idx],predictions[idx],target[idx],dataset[idx],path[idx])


        lossAvg /= len(iterator)

        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')

        if table:
            wandb.log({prefix+"table":test_table})
            matrix=wandb.plot.confusion_matrix(probs=None,
                                        y_true=all_targets, preds=all_predictions,
                                        class_names=["Full","Incomplete"])

            wandb.log({prefix+"confussion_matrix":matrix})
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            prefix + "eval_loss": lossAvg.cpu().numpy(),
            prefix + "eval_accuracy": accuracy,
            prefix + "eval_precision": precision,
            prefix + "eval_recall": recall,
            prefix + "eval_f1": f1
        })


