import logging

import PIL
import torch.utils.data as td
import tqdm
from PIL import Image

logger = logging.getLogger('iCARL')


class myLoader(td.Dataset):
    def __init__(self, data, transform=None, cuda=False):
        self.data = data

        self.transform = transform
        self.cuda = cuda
        self.len = len(data[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < len(self.data[0]))
        assert (index < self.len)
        img = Image.open(self.data[0][index])
        target = self.data[1][index]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

class myLoaderRAM(td.Dataset):
    def __init__(self, data, transform=None, cuda=False):
        self.data = data

        self.transform = transform
        self.cuda = cuda
        self.len = len(data[0])
        self.loadInRam()

    def loadInRam(self):
        self.loaded_data = []
        for i in tqdm.tqdm(self.data[0]):
            img = Image.open(i)
            if self.transform is not None:
                img = self.transform(img)
            self.loaded_data.append(img)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < len(self.data[0]))
        assert (index < self.len)
        target = self.data[1][index]
        img = self.loaded_data[index]
        return img, target



class myLoaderDecoded(td.Dataset):
    '''
    This loader class decodes all the images into tensors; this removes the decoding time.
    '''

    def __init__(self, data, transform=None, cuda=False):

        self.data = data

        self.transform = transform
        self.cuda = cuda
        self.len = len(data)
        self.decodeImages()

    def decodeImages(self):
        self.loaded_data = []
        logger.info("Resizing Images")
        for i in tqdm.tqdm(self.data):
            i = i[0]
            img = Image.open(i)
            img = img.resize((300, 300), PIL.Image.ANTIALIAS)
            img.save(i)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < len(self.data))
        assert (index < self.len)

        img = Image.open(self.data[index][0])
        target = self.data[index][1]
        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    pass
