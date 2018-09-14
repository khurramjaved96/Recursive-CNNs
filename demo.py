''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import cv2
import numpy as np

import evaluation


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="../output.jpg", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default="../cornerModelWell")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="../documentModelWell")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_processor()

    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel)

    img = cv2.imread(args.imagePath)

    oImg = img

    extracted_corners = corners_extractor.get(oImg)
    corner_address = []
    # Refine the detected corners using corner refiner
    image_name = 0
    for corner in extracted_corners:
        image_name += 1
        corner_img = corner[0]
        refined_corner = np.array(corner_refiner.get_location(corner_img, 0.85))

        # Converting from local co-ordinate to global co-ordinates of the image
        refined_corner[0] += corner[1]
        refined_corner[1] += corner[2]

        # Final results
        corner_address.append(refined_corner)

    for a in range(0, len(extracted_corners)):
        cv2.line(oImg, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)

    cv2.imwrite(args.outputPath, oImg)
