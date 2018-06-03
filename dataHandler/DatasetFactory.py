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
    def get_dataset(directory, type="document"):
        if type=="document":
            return data.SmartDoc(directory)
        elif type =="corner":
            return data.SmartDocCorner(directory)
