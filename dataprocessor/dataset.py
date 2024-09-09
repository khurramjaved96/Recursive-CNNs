''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import csv
import logging
import os
import xml.etree.ElementTree as ET
import pandas as pd

import numpy as np
from torchvision import transforms

import utils.utils as utils

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

logger = logging.getLogger('iCARL')





class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []


class CompleteDocuments(Dataset):
    def __init__(self,directory="data",csv_name="gt.csv"):
        super().__init__("completedocuments")
        self.csv_name=csv_name
        self.data = []
        self.labels = []
        self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                   transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                                                   transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                  transforms.ToTensor()])
        paths = []
        labels = []
        directories = []
        for d in directory:
            self.directory = d
            df=pd.read_csv(os.path.join(d,csv_name))
            arr = df["label"]
            encoded_arr = np.zeros((arr.size, arr.max() + 1), dtype=int)
            encoded_arr[np.arange(arr.size), arr] = 1
            df["label"] = encoded_arr.tolist()


            for idx,row in df.iterrows():
                paths.append(row["file_path"])
                labels.append(row["label"])
                directories.append(row["directory"])
        self.data=paths
        self.labels=labels
        self.myData = [self.data, self.labels,directories]



class SmartDoc(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data",csv_name="gt.csv"):
        super().__init__("smartdoc")
        self.csv_name=csv_name
        self.data = []
        self.labels = []
        for d in directory:
            self.directory = d
            self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                       transforms.ColorJitter(.5, .5, .5, 0.2),
                                                       transforms.ToTensor()])

            self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                      transforms.ToTensor()])

            logger.info("Pass train/test data paths here")

            self.classes_list = {}

            file_names = []
            print (os.path.join(self.directory, self.csv_name))
            with open(os.path.join(self.directory, self.csv_name), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[0])
                    # print(row[0])
                    self.data.append(os.path.join(self.directory, row[0].replace("\\\\","/")))
                    test = row[1].replace("array", "").replace("\\","").replace("\\\\","/")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


class SmartDocDirectories(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                for file in os.listdir(directory + "/" + folder):
                    images_dir = directory + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file.replace(".avi",".gt").replace("\\","/") + ".xml")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                # print(im_no)
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                # img = cv2.imread(images_dir + "/" + image)
                                self.data.append(os.path.join(images_dir, image))

                                for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (
                                        int(float(myDict['x'])), int(float(myDict['y'])))

                                ground_truth = np.asarray(
                                    (list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                     list_of_points["bl"]))
                                ground_truth = utils.sort_gt(ground_truth)
                                self.labels.append(ground_truth)

        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

class SelfCollectedDataset(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for image in os.listdir(directory):
            # print (image)
            if image.endswith("jpg") or image.endswith("JPG"):
                if os.path.isfile(os.path.join(directory, image + ".csv")):
                    with open(os.path.join(directory, image + ".csv"), 'r') as csvfile:
                        spamwriter = csv.reader(csvfile, delimiter=' ',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

                        img_path = os.path.join(directory, image)

                        gt = []
                        for row in spamwriter:
                            gt.append(row)
                        gt = np.array(gt).astype(np.float32)
                        ground_truth = utils.sort_gt(gt)
                        self.labels.append(ground_truth)
                        self.data.append(img_path)

        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])




class SmartDocCorner(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        for d in directory:
            self.directory = d
            self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                       transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                                       transforms.ToTensor()])

            self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                      transforms.ToTensor()])

            logger.info("Pass train/test data paths here")

            self.classes_list = {}

            file_names = []
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    if len(row)>0:
                        file_names.append(row[0])
                        self.data.append(os.path.join(self.directory, row[0]))
                        test = row[1].replace("array", "")
                        self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 2))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]
