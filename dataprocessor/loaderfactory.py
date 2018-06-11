''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''



import dataprocessor.dataloaders as loader


class LoaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_loader(type, data, transform=None, cuda=False):
        if type=="hdd":
            return loader.HddLoader(data, transform=transform,
                                    cuda=cuda)
        elif type =="ram":
            return loader.RamLoader(data, transform=transform,
                                    cuda=cuda)