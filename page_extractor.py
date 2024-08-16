import cv2
import matplotlib.pyplot as plt
import numpy as np

import evaluation


class PageExtractor(object):
    def __init__(self, cornerModel_path: str, documentModel_path: str):
        self.corners_extractor = evaluation.corner_extractor.GetCorners(documentModel_path)
        self.corner_refiner = evaluation.corner_refiner.corner_finder(cornerModel_path)

    def extract_corners(self, image_path: str, retain_factor: float=.85):
        img = cv2.imread(image_path)

        oImg = img
        self.img = oImg

        extracted_corners = self.corners_extractor.get(oImg)
        # plt.imshow(oImg)
        # plt.show()
        corner_address = []
        # Refine the detected corners using corner refiner
        image_name = 0
        for corner in extracted_corners:
            image_name += 1
            corner_img = corner[0]
            refined_corner = np.array(self.corner_refiner.get_location(corner_img, float(retain_factor)))

            # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[0] += corner[1]
            refined_corner[1] += corner[2]

            # Final results
            corner_address.append(refined_corner)
        return corner_address

    def highlight_bounding_box(self, image_path: str, retain_factor: float=.85):

        corners = self.extract_corners(image_path, retain_factor)
        for a in range(0, len(corners)):
            cv2.line(self.img, tuple(corners[a % 4]), tuple(corners[(a + 1) % 4]), (255, 0, 0), 4)
        return corners, self.img

    def extract_document(self, image_path: str, retain_factor: float):
        corners = self.extract_corners(image_path, retain_factor)




        corners = [corners[i] for i in [0, 3, 2, 1]]
        # corners = [[entry[1],entry[0]] for entry in corners]
        (tl, tr, br, bl) = corners
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],  # Top left point
            [maxWidth , 0],  # Top right point
            [maxWidth , maxHeight ],  # Bottom right point
            [0, maxHeight ]],  # Bottom left point
            dtype="float32"  # Date type
        )
        # tl, tr, br, bl
        # corners=[corners[i] for i in [0,3,1,2]]
        # corners=[[corners[i][1],corners[i][0
        # ]] for i in [0,3,1,2]]

        corners = np.array([tl,tr,br,bl], dtype="float32")

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))


        return corners, warped

# extractor=PageExtractor(r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch",r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\documentModelPyTorch")
# result=extractor.extract_document(r"C:\Users\isaac\PycharmProjects\document_localization\kosmos-dataset\20230313_141209.jpg",.85)