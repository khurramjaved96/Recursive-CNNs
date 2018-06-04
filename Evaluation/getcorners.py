import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import model


class GetCorners:
    def __init__(self, checkpoint_dir="../documentModel"):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()

    def get(self, img):

        import time
        start = time.time()
        myImage = np.copy(img)
        img = Image.fromarray(img)
        test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                             transforms.ToTensor()])
        img_temp = test_transform(img)

        img_temp = img_temp.unsqueeze(0)
        if torch.cuda.is_available():
            img_temp = img_temp.cuda()

        response = self.model(Variable(img_temp)).cpu().data.numpy()[0]

        response = np.array(response)
        #
        # x = response[[0, 2, 4, 6]]
        # y = response[[1, 3, 5, 7]]

        if response[0]<response[6]:
            x = response[[0, 6, 4, 2]]
            y = response[[1, 7, 5, 3]]
        else:
            x = response[[0, 2, 4, 6]]
            y = response[[1, 3, 5, 7]]

        x = x * myImage.shape[1]
        y = y * myImage.shape[0]
        # print(x, y)

        tl = myImage[max(0, int(2 * y[0] - (y[3] + y[0]) / 2)):int((y[3] + y[0]) / 2),
             max(0, int(2 * x[0] - (x[1] + x[0]) / 2)):int((x[1] + x[0]) / 2)]

        tr = myImage[max(0, int(2 * y[1] - (y[1] + y[2]) / 2)):int((y[1] + y[2]) / 2),
             int((x[1] + x[0]) / 2):min(myImage.shape[1] - 1, int(x[1] + (x[1] - x[0]) / 2))]

        br = myImage[int((y[1] + y[2]) / 2):min(myImage.shape[0] - 1, int(y[2] + (y[2] - y[1]) / 2)),
             int((x[2] + x[3]) / 2):min(myImage.shape[1] - 1, int(x[2] + (x[2] - x[3]) / 2))]

        bl = myImage[int((y[0] + y[3]) / 2):min(myImage.shape[0] - 1, int(y[3] + (y[3] - y[0]) / 2)),
             max(0, int(2 * x[3] - (x[2] + x[3]) / 2)):int((x[3] + x[2]) / 2)]

        tl = (tl, max(0, int(2 * x[0] - (x[1] + x[0]) / 2)), max(0, int(2 * y[0] - (y[3] + y[0]) / 2)))
        tr = (tr, int((x[1] + x[0]) / 2), max(0, int(2 * y[1] - (y[1] + y[2]) / 2)))
        br = (br, int((x[2] + x[3]) / 2), int((y[1] + y[2]) / 2))
        bl = (bl, max(0, int(2 * x[3] - (x[2] + x[3]) / 2)), int((y[0] + y[3]) / 2))
        end = time.time()
        # print("Time to Extract Corners", start - end)
        return tl, tr, br, bl
