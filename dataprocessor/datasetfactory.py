''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import dataprocessor.dataset as data
import torchvision

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(directory, type="document",csv_name="gt.csv"):
        if type=="document":
            return data.SmartDoc(directory,csv_name)
        elif type =="corner":
            return data.SmartDocCorner(directory)
        elif type=="CIFAR":
            return torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        elif "complete_document":
            return data.CompleteDocuments(directory, csv_name)