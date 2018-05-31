''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import torchvision.models as models
import model.resnet32 as resnet
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type):
        if model_type == "resnet":
            return resnet.resnet20(8)


        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
