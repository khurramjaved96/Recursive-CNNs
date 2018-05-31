import numpy as np

from utils import utils


def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputDir", help="output Directory of Data")
    parser.add_argument("-i", "--inputDir", help="input Directory of data")
    parser.add_argument("-s", "--saveName", help="fileNameForSaving")
    return  parser.parse_args()

if __name__ == "__main__":
    args  = argsProcessor()

    inputDataDir = args.inputDir
    outputDataDir = args.outputDir
    GT_DIR = inputDataDir + "/gt.csv"
    VALIDATION_PERCENTAGE = .20
    TEST_PERCENTAGE = .10
    Debug = True
    size= (32,32)

    image_list, gt_list, file_name = utils.load_data(inputDataDir, GT_DIR, size=size, debug=Debug, limit=10000)
    image_list, gt_list = utils.unison_shuffled_copies(image_list, gt_list)


    print (len(image_list))


    if (Debug):
        print ("(Image_list_len, gt_list_len)", (len(image_list), len(gt_list)))
    train_image = image_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
    validate_image = image_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]

    train_gt = gt_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
    validate_gt = gt_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]
    if (Debug):
        print ("(Train_Image_len, Train_gt_len)", (len(train_image), len(train_gt)))
        print ("(Validate_Image_len, Validate_gt_len)", (len(validate_image), len(validate_gt)))

    np.save(outputDataDir + args.saveName + "trainGtCorners", train_gt)
    np.save(outputDataDir + args.saveName + "trainImagesCorners", train_image)
    np.save(outputDataDir + args.saveName + "validateGTCorners", validate_gt)
    np.save(outputDataDir + args.saveName + "validateImagesCorners", validate_image)
    # 0/0