import json
import os
import random
import matplotlib.pyplot as plt
import wandb
from scipy.stats import bernoulli
from PIL import Image
import numpy as np

from label_studio_dataset import LabelStudioDataset
from document_localization_metrics import DocumentLocalizationMetrics
from document_localization_utils import DocumentVisualization
from page_extractor import PageExtractor

dataset = LabelStudioDataset(
    r"C:\Users\isaac\PycharmProjects\document_localization\meta_files\cleaned_data.json",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\kosmos-dataset")

extractor = PageExtractor(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\documentModelPyTorch")


# %%
def _area(rect):
    x, y = zip(*rect)
    width = max(x) - min(x)
    height = max(y) - min(y)
    return width * height


IoUs = []
recalls = []
precisions = []
distance = []
json_contents = json.load(open(r"C:\Users\isaac\PycharmProjects\document_localization\meta_files\cleaned_data.json"))

# f=dataset[223]

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="document-detection",
#     config={"Dataset":"Kosmosv0",
#             "CornerModel":"Pre-trained",
#             "DocumentModel":"Pre-trained"})
# wandb.run.name="Pre-trained baseline"
# # %%
# wandb_table=wandb.Table(columns=["path","Pred VS Label ","extracted_img","IoU","Recall","Precision"])
for index, (img_path, corners, dimensions) in enumerate(dataset):

    # print(index)

    corners = [tuple(corners[key]) for key in corners.keys()]

    try:
    # bool(bernoulli(.06).rvs(1)[0])
        if bool(bernoulli(.06).rvs(1)[0]):
            rect,warped = extractor.extract_document(img_path, .85)
            dims = extractor.img.shape
            rect=[(cordinate[1]/dims[0],cordinate[0]/dims[1]) for cordinate in rect]

            original_img=Image.fromarray(extractor.img)
            original_img=original_img.resize((int(dims[1]/10),int(dims[0]/10)))
            original_img=DocumentVisualization.graph_label_vs_pred(original_img,rect,corners)


            warped = Image.fromarray(warped)
            warped = warped.resize((int(dims[1] / 10), int(dims[0] / 10)))



            Iou, recall, precision = DocumentLocalizationMetrics().calculate_metrics(rect, corners)
            IoUs.append(Iou)
            recalls.append(recall)
            precisions.append(precision)
            # wandb_table.add_data(os.path.basename(img_path),wandb.Image(np.array(original_img)),wandb.Image(np.array(warped)),Iou,recall,precision)

        else:
            rect = extractor.extract_corners(img_path, .85)
            dims = extractor.img.shape
            rect = [(cordinate[1] / dims[0], cordinate[0] / dims[1])  for cordinate in rect]

            Iou, recall, precision = DocumentLocalizationMetrics().calculate_metrics(rect, corners)
            IoUs.append(Iou)
            recalls.append(recall)
            precisions.append(precision)
    except:
        pass
        # IoUs.append(0)
        # recalls.append(0)
        # precisions.append(0)
import numpy as np

wandb.log({"Sample table":wandb_table})

plt.hist(IoUs, bins=20, )
plt.title("IoU Histogram")
plt.show()
print("Iou (accuracy:)", np.round(np.array(IoUs).mean() * 100, 2), "%")
print("Recall:", np.round(np.array(recalls).mean() * 100, 2), "%")
print("Precision:", np.round(np.array(precisions).mean() * 100, 2), "%")


wandb.log({"IoU (accuracy)": np.array(IoUs).mean()})

wandb.log({"Recall": np.array(recalls).mean()})

wandb.log({"Precision:": np.array(precisions).mean()})
plt.hist(recalls, bins=20, )
plt.title("Recall Histogram")
plt.show()
plt.hist(precisions, bins=20, )
plt.title("Precision Histogram")
plt.show()

iou_hist = np.histogram(IoUs)
wandb.log({"IoU hist": wandb.Histogram(np_histogram=iou_hist)})

iou_hist = np.histogram(recalls)
wandb.log({"Recall hist": wandb.Histogram(np_histogram=iou_hist)})

iou_hist = np.histogram(precisions)
wandb.log({"Precision hist": wandb.Histogram(np_histogram=iou_hist)})



wandb.finish()