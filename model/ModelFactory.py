''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import torchvision.models as models
import model.resnet32 as resnet
import model.testModel as tm
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset):
        if model_type == "resnet":
            if dataset=='document':
                return resnet.resnet20(8)
            elif dataset=='corner':
                return resnet.resnet20(2)
        elif model_type == 'standard':
            return tm.Net(8)
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
