''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import dataHandler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(directory):
        return data.SmartDoc(directory)
