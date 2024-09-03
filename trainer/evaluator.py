''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging
from typing import Dict, List, Tuple, Union
from sys import prefix

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn.functional as F
from jedi.inference.gradual.typeshed import try_to_load_stub_cached

import torch
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


logger = logging.getLogger('iCARL')
from PIL import ImageDraw
def draw_multiple_bounding_boxes(image, bounding_boxes):
    """
    Draws multiple bounding boxes on an image.

    :param image_path: Path to the input image.
    :param bounding_boxes: List of bounding boxes, each defined as [lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x].
    :param output_path: Path to save the output image.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each bounding box and draw it on the image
    for bounding_box in bounding_boxes:
        lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x = bounding_box
        rect_coords = [lower_bound_x*200, lower_bound_y*200, upper_bound_x*200, upper_bound_y*200]
        try:
            draw.rectangle(rect_coords, outline="red", width=1)
        except:
            pass
    return image


def highlight_coordinates(image, coordinates,color,  radius=1):
    """
    Highlights coordinates on an image by drawing small green circles around them.

    :param image_path: Path to the input image.
    :param coordinates: List of coordinates, each defined as (x, y).
    :param output_path: Path to save the output image.
    :param radius: Radius of the circle used to highlight the coordinates. Default is 5.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each coordinate and draw a small circle around it
    for (x, y) in coordinates:
        left_up_point = ((x*200 - radius), (y*200 - radius))
        right_down_point = ((x*200 + radius), (y*200 + radius))
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2, fill=color)
    return image


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
        self.table=wandb.Table(columns=["img","tl","tr","br","bl","path","total"])

    def cordinate_within_intervals(self, cordinate, x_interval, y_interval) -> int:

        is_within_x = (x_interval[0] <= cordinate[0] <= x_interval[1])
        is_within_y = (y_interval[0] <= cordinate[1] <= y_interval[1])

        return int(is_within_x and is_within_y)


    def fill_table(self,imgs,results):



        for idx in range(len(imgs)):
            img=imgs[idx].cpu().data.numpy()
            img= np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img=Image.fromarray(img).resize((200,200))
            # img=np.array(img)
            result=results[idx]

            bb_s=[result["top_left"][2:],
                  result["top_right"][2:],
                  result["bottom_right"][2:],
                  result["bottom_left"][2:]]

            cordinates=[result["top_left"][0],
                        result["top_right"][0],
                        result["bottom_right"][0],
                        result["bottom_left"][0]]
            labels=[result["top_left"][1],
                        result["top_right"][1],
                        result["bottom_right"][1],
                        result["bottom_left"][1]]

            # for bb_ in bb_s:
            img=draw_multiple_bounding_boxes(img,bb_s)
            img=highlight_coordinates(img,cordinates,"green")
            img=highlight_coordinates(img,labels,"blue")


            path=result["path"]
            contains_tl=result["contains_tl"]
            contains_tr=result["contains_tr"]
            contains_br=result["contains_br"]
            contains_bl=result["contains_bl"]
            total=result["total_corners"]
            self.table.add_data(wandb.Image(np.array(img)),contains_tl,contains_tr,contains_br,contains_bl,path,total)

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

            top_left_pred=(x_cords[entry,0],y_cords[entry,0])
            top_right_pred=(x_cords[entry,1],y_cords[entry,1])
            bottom_right_pred=(x_cords[entry,2],y_cords[entry,2])
            bottom_left_pred=(x_cords[entry,3],y_cords[entry,3])

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

                "top_left": (top_left_pred,top_left,
                             top_left_y_lower_bound,
                             top_left_y_upper_bound,
                             top_left_x_lower_bound,
                             top_left_x_upper_bound),
                "top_right": (top_right_pred,top_right,
                              top_right_y_lower_bound,
                              top_right_y_upper_bound,
                              top_right_x_lower_bound,
                              top_right_x_upper_bound),
                "bottom_right": (bottom_right_pred,bottom_right,
                                 bottom_right_y_lower_bound,
                                 bottom_right_y_upper_bound,
                                 bottom_right_x_lower_bound,
                                 bottom_right_x_upper_bound),
                "bottom_left": (bottom_left_pred,bottom_left,
                                bottom_left_y_lower_bound,
                                bottom_left_y_upper_bound,
                                bottom_left_x_lower_bound,
                                bottom_left_x_upper_bound),
            }
            resut_dicts.append(resut_dict)
        return resut_dicts

    def evaluate(self, model, iterator, epoch,prefix,table):
        model.eval()
        lossAvg = None
        classification_results=[]
        with torch.no_grad():
            for img, target,paths in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))

                loss = F.mse_loss(response, Variable(target.float()))
                loss = torch.sqrt(loss)

                # model_prediction = self.model(img_temp)[0]

                model_prediction = np.array(response.cpu().data.numpy())

                x_cords = model_prediction[:, [0, 2, 4, 6]]
                y_cords = model_prediction[:, [1, 3, 5, 7]]

                classification_result = self.evaluate_corners(x_cords, y_cords, target,paths)
                classification_results.extend(classification_result)
                if table:
                    self.fill_table(img,classification_result)

                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))
        df=pd.DataFrame(classification_results)

        df.to_csv(r"/home/ubuntu/document_localization/Recursive-CNNs/predictions.csv")

        lossAvg /= len(iterator)
        total_corners=df["total_corners"]
        wandb.log({"epoch": epoch,
                   prefix+"eval_loss": lossAvg,
                   prefix+"accuracy": lossAvg,
                   prefix+"4_corners_accuracy": np.sum(total_corners>=3)/len(total_corners),
                   prefix+"3_corners_accuracy": np.sum(total_corners==4)/len(total_corners),

                   })
        # logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))
        if table:
            wandb.log({prefix+"table":self.table})




class CompleteDocEvaluator():
    '''
    Evaluator class for softmax classification
    '''

    def __init__(self, cuda):
        self.cuda = cuda

    def evaluate(self, model, iterator, epoch, prefix, table=True):
        model.eval()

        test_table = wandb.Table(columns=["img", "directory", "prediction", "label", "dataset", "path"])

        lossAvg = None
        all_targets = []
        all_predictions = []
        loss_function = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for img, target, directory, dataset, path in tqdm(iterator):
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
                img = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
                if table:
                    for idx in range(len(target)):
                        numpy_img = wandb.Image(img[idx])
                        test_table.add_data(numpy_img, directory[idx], predictions[idx], target[idx], dataset[idx],
                                            path[idx])

        lossAvg /= len(iterator)

        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')

        if table:
            wandb.log({prefix + "table": test_table})
            matrix = wandb.plot.confusion_matrix(probs=None,
                                                 y_true=all_targets, preds=all_predictions,
                                                 class_names=["Full", "Incomplete"])

            wandb.log({prefix + "confussion_matrix": matrix})
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            prefix + "eval_loss": lossAvg.cpu().numpy(),
            prefix + "eval_accuracy": accuracy,
            prefix + "eval_precision": precision,
            prefix + "eval_recall": recall,
            prefix + "eval_f1": f1
        })
