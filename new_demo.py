from page_extractor import PageExtractor
from PIL import Image
import matplotlib.pyplot as plt

import os
extractor = PageExtractor(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\documentModelPyTorch")


img_path=r"C:\Users\isaac\PycharmProjects\document_localization\ine_examles"
for img in os.listdir(img_path):
    corner,img=extractor.highlight_bounding_box(os.path.join(img_path,img),.85)
    plt.imshow(img)
    plt.show()
