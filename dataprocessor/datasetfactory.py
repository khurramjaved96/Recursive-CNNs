''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import dataprocessor.dataset as data
import torchvision

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(directory, type="document"):
        if type=="document":
            return data.SmartDoc(directory)
        elif type =="corner":
            return data.SmartDocCorner(directory)
        elif type=="CIFAR":
            return torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
