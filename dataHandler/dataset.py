''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import csv
import logging
import os

import numpy as np
from torchvision import transforms

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

logger = logging.getLogger('iCARL')


class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, name):
        self.name = name
        self.data = None
        self.labels = None


class SmartDoc(Dataset):
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
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


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
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 2))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


