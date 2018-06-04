import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import model


def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="../img.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="../output.jpg", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.95")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default="./TrainedModel/cornerRefiner.pb")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="./TrainedModel/getCorners.pb")
    return parser.parse_args()


def refineCorner(img, model, retainFactor):
    import time

    start = time.time()
    ans_x = 0.0
    ans_y = 0.0

    o_img = np.copy(img)

    y = None
    x_start = 0
    y_start = 0
    up_scale_factor = (img.shape[1], img.shape[0])

    myImage = np.copy(o_img)

    test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                         transforms.ToTensor()])

    CROP_FRAC = retainFactor
    while (myImage.shape[0] > 10 and myImage.shape[1] > 10):

        # img_temp = cv2.resize(myImage, (32, 32))
        img_temp = Image.fromarray(myImage)
        img_temp = test_transform(img_temp)
        img_temp = img_temp.unsqueeze(0)

        response = model(Variable(img_temp)).cpu().data.numpy()
        response = response[0]
        # response = np.array([0.5, 0.5])
        response_up = response

        response_up = response_up * up_scale_factor
        # response_up = np.array(response_up[1], response_up[0])
        y = response_up + (x_start, y_start)
        x_loc = int(y[0])
        y_loc = int(y[1])

        if x_loc > myImage.shape[1] / 2:
            start_x = min(x_loc + int(round(myImage.shape[1] * CROP_FRAC / 2)), myImage.shape[1]) - int(round(
                myImage.shape[1] * CROP_FRAC))
        else:
            start_x = max(x_loc - int(myImage.shape[1] * CROP_FRAC / 2), 0)
        if y_loc > myImage.shape[0] / 2:
            start_y = min(y_loc + int(myImage.shape[0] * CROP_FRAC / 2), myImage.shape[0]) - int(
                myImage.shape[0] * CROP_FRAC)
        else:
            start_y = max(y_loc - int(myImage.shape[0] * CROP_FRAC / 2), 0)

        ans_x += start_x
        ans_y += start_y

        myImage = myImage[start_y:start_y + int(myImage.shape[0] * CROP_FRAC),
                  start_x:start_x + int(myImage.shape[1] * CROP_FRAC)]
        img = img[start_y:start_y + int(img.shape[0] * CROP_FRAC), start_x:start_x + int(img.shape[1] * CROP_FRAC)]
        up_scale_factor = (img.shape[1], img.shape[0])

    ans_x += y[0]
    ans_y += y[1]
    end = time.time()
    print("Time to refine corners", end - start)

    return (int(round(ans_x)), int(round(ans_y)))


def getCorners(img, model, o_img):
    import time
    start = time.time()
    myImage = np.copy(o_img)

    test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                         transforms.ToTensor()])

    img_temp = test_transform(img)

    img_temp = img_temp.unsqueeze(0)
    if torch.cuda.is_available():
        img_temp = img_temp.cuda()

    response = model(Variable(img_temp)).cpu().data.numpy()[0]

    response = np.array(response)
    print(response)

    x = response[[0, 6, 4, 2]]
    y = response[[1, 7, 5, 3]]
    x = x * myImage.shape[1]
    y = y * myImage.shape[0]
    print(x, y)

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
    print("Time to Extract Corners", start - end)
    return tl, tr, br, bl


if __name__ == "__main__":
    args = argsProcessor()

    docDetector = model.ModelFactory.get_model("resnet", 'document')
    corDetector = model.ModelFactory.get_model("resnet", 'corner')

    docDetector.load_state_dict(torch.load("../documentModel", map_location='cpu'))
    corDetector.load_state_dict(torch.load("../cornerModel", map_location='cpu'))

    img = Image.open(args.imagePath)

    oImg = np.array(img)
    print(oImg.shape)

    if torch.cuda.is_available():
        docDetector.cuda()
        corDetector.cuda()

    data = getCorners(img, docDetector, oImg)

    corner_address = []
    counter = 0
    for b in data:
        a = b[0]
        print("CORENR SHAPE", a.shape)
        cv2.imwrite("../TempImg" + str(counter) + ".jpg", a)
        # cv2.waitKey(50)
        temp = np.array(refineCorner(a, corDetector, float(args.retainFactor)))
        temp[0] += b[1]
        temp[1] += b[2]
        corner_address.append(temp)
        print(temp)
        counter += 1

    for a in range(0, len(data)):
        cv2.line(oImg, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)

    cv2.imwrite(args.outputPath, oImg)
    #
    #
