import cv2
import matplotlib.pyplot as plt
import numpy as np
import dataprocessor
import evaluation
from PIL import Image
from document_localization_metrics import DocumentLocalizationMetrics
import pandas as pd
cornerModel_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch"
documentModel_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\experiments882024\Validation_run_12\Validation_rundocument_resnet.pb"
corners_extractor = evaluation.corner_extractor.GetCorners(documentModel_path)
corner_refiner = evaluation.corner_refiner.corner_finder(cornerModel_path)
from page_extractor import PageExtractor
from document_localization_utils import DocumentVisualization

extractor_object = PageExtractor(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\documentModelPyTorch")

dataset = dataprocessor.DatasetFactory.get_dataset(
    [r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train"],
    "document",
    "test.csv")

corner_name = ["top_left",
               "bottom_left",
               "bottom_right",

               "top_right", ]
path=[]

top_left_interval_x_s = []
bottom_left_interval_x_s = []
bottom_right_interval_x_s = []
top_right_interval_x_s = []

top_left_interval_y_s = []
bottom_left_interval_y_s = []
bottom_right_interval_y_s = []
top_right_interval_y_s = []

top_left_second_phase = []
bottom_left_second_phase = []
bottom_right_second_phase = []
top_right_second_phase = []

top_left_second_phase_inside = []
bottom_left_second_phase_inside = []
bottom_right_second_phase_inside = []
top_right_second_phase_inside = []


IoUs=[]
recalls=[]
precisions=[]

def cordinate_within_intervals(cordinate,x_interval, y_interval):

    is_within_x=(x_interval[0]<=cordinate[0]<=x_interval[1])
    is_within_y=(y_interval[0]<=cordinate[1]<=y_interval[1])

    return (is_within_x and is_within_y)

def append_bounding_boxes(extracted_corners,labeled_corners):

    top_left=labeled_corners[0]
    top_right=labeled_corners[1]
    bottom_right=labeled_corners[2]
    bottom_left=labeled_corners[3]


    top_left_interval_x=(extracted_corners[0][1], extracted_corners[0][3])
    bottom_left_interval_x=(extracted_corners[3][1], extracted_corners[3][3])
    bottom_right_interval_x=(extracted_corners[2][1], extracted_corners[2][3])
    top_right_interval_x=(extracted_corners[1][1], extracted_corners[1][3])

    top_left_interval_y=(extracted_corners[0][2], extracted_corners[0][4])
    bottom_left_interval_y=(extracted_corners[3][2], extracted_corners[3][4])
    bottom_right_interval_y=(extracted_corners[2][2], extracted_corners[2][4])
    top_right_interval_y=(extracted_corners[1][2], extracted_corners[1][4])

    top_left_interval_x_s.append(top_left_interval_x)
    bottom_left_interval_x_s.append(bottom_left_interval_x)
    bottom_right_interval_x_s.append(bottom_right_interval_x)
    top_right_interval_x_s.append(top_right_interval_x)
    top_left_interval_y_s.append(top_left_interval_y)
    bottom_left_interval_y_s.append(bottom_left_interval_y)
    bottom_right_interval_y_s.append(bottom_right_interval_y)
    top_right_interval_y_s.append(top_right_interval_y)



    top_left_second_phase.append((extracted_corners[0][0].size != 0))
    bottom_left_second_phase.append((extracted_corners[3][0].size != 0))
    bottom_right_second_phase.append((extracted_corners[2][0].size != 0))
    top_right_second_phase.append((extracted_corners[1][0].size != 0))

    top_left_second_phase_inside.append(cordinate_within_intervals(top_left,top_left_interval_x,top_left_interval_y))
    top_right_second_phase_inside.append(cordinate_within_intervals(top_right,top_right_interval_x,top_right_interval_y))
    bottom_right_second_phase_inside.append(cordinate_within_intervals(bottom_right,bottom_right_interval_x,bottom_right_interval_y))
    bottom_left_second_phase_inside.append(cordinate_within_intervals(bottom_left,bottom_left_interval_x,bottom_left_interval_y))





# top_left_error_distance = []
# bottom_left_error_distance = []
# bottom_right_error_distance = []
# top_right_error_distance = []



# %%

path=path[:3975]
top_left_interval_x_s=top_left_interval_x_s[:3975]
bottom_left_interval_x_s=bottom_left_interval_x_s[:3975]
bottom_right_interval_x_s=bottom_right_interval_x_s[:3975]
top_right_interval_x_s=top_right_interval_x_s[:3975]
top_left_interval_y_s=top_left_interval_y_s[:3975]
bottom_left_interval_y_s=bottom_left_interval_y_s[:3975]
bottom_right_interval_y_s=bottom_right_interval_y_s[:3975]
top_right_interval_y_s=top_right_interval_y_s[:3975]
top_left_second_phase=top_left_second_phase[:3975]
bottom_left_second_phase=bottom_left_second_phase[:3975]
bottom_right_second_phase=bottom_right_second_phase[:3975]
top_right_second_phase=top_right_second_phase[:3975]
top_left_second_phase_inside=top_left_second_phase_inside[:3975]
bottom_left_second_phase_inside=bottom_left_second_phase_inside[:3975]
bottom_right_second_phase_inside=bottom_right_second_phase_inside[:3975]
top_right_second_phase_inside=top_right_second_phase_inside[:3975]
IoUs=IoUs[:3975]
recalls=recalls[:3975]
precisions=precisions[:3975]
#%%
for img_path, label in zip(dataset.myData[0][3978:], dataset.myData[1][3978:]):

    path.append(img_path)

    # _,warped=extractor_object.extract_document(img_path, .85)
    #
    # plt.imshow(warped)
    # plt.show()

    img = cv2.imread(img_path)
    # label=label.reshape((4,2))

    x_cords = label[[0, 2, 4, 6]] * img.shape[1]
    y_cords = label[[1, 3, 5, 7]] * img.shape[0]


    label=np.array([x_cords,y_cords]).T

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # ax.scatter(x_cords,y_cords  )
    # for index in range(len(x_cords)):
    #     ax.text(x_cords[index], y_cords[index], ["tl", "tr", "br", "bl"][index])
    # plt.show()
    oImg = img.copy()

    extracted_corners = corners_extractor.get(oImg, True)

    append_bounding_boxes(extracted_corners,label)

    corner_address = []
    # Refine the detected corners using corner refiner
    image_name = 0
    for corner in extracted_corners:
        corner_img = corner[0]

        image_name += 1
        refined_corner = np.array(corner_refiner.get_location(corner_img, float(.85)))

        # Converting from local co-ordinate to global co-ordinates of the image
        refined_corner[0] += corner[1]
        refined_corner[1] += corner[2]

        # Final results
        corner_address.append(refined_corner)
    # rect = [(cordinate[1] / img.shape[0], cordinate[0] / img.shape[1]) for cordinate in corner_address]
    rect = [(cordinate[0]/img.shape[1] , cordinate[1]/img.shape[0] ) for cordinate in corner_address]

    label=[(entry[0]/img.shape[1],entry[1]/img.shape[0] ) for entry in label]

    # x_rect=[entry[0] for entry in rect]
    # y_rect=[entry[1] for entry in rect]
    # x_label=[entry[0] for entry in label ]
    # y_label=[entry[1] for entry in label ]
    # plt.imshow(img)
    # plt.scatter(x_rect,y_rect, color="red")
    # plt.scatter(x_label, y_label,color="blue")
    # plt.show()
    try:
        Iou, recall, precision = DocumentLocalizationMetrics().calculate_metrics(rect, label)
        IoUs.append(Iou)
        recalls.append(recall)
        precisions.append(precision)
    except:
        IoUs.append(0)
        recalls.append(0)
        precisions.append(0)
    # rect, warped = extractor_object.extract_document(img_path, .85)
    # dims = extractor_object.img.shape
    # rect = [(cordinate[1] / dims[0], cordinate[0] / dims[1]) for cordinate in rect]
    #
    # original_img = Image.fromarray(extractor_object.img)
    # original_img = original_img.resize((int(dims[1] / 10), int(dims[0] / 10)))
    # original_img = DocumentVisualization.graph_bb(original_img, rect)
    # # # original_img = DocumentVisualization.graph_label_vs_pred(original_img, rect, corners)
    # # # plt.imshow(warped)
    # # # plt.show()
    # # plt.imshow(original_img)
    # # plt.show()
#%%
df=pd.DataFrame({
    "path": path,
    "top_left_interval_x_s": top_left_interval_x_s,
    "bottom_left_interval_x_s": bottom_left_interval_x_s,
    "bottom_right_interval_x_s": bottom_right_interval_x_s,
    "top_right_interval_x_s": top_right_interval_x_s,
    "top_left_interval_y_s": top_left_interval_y_s,
    "bottom_left_interval_y_s": bottom_left_interval_y_s,
    "bottom_right_interval_y_s": bottom_right_interval_y_s,
    "top_right_interval_y_s": top_right_interval_y_s,
    "top_left_second_phase": top_left_second_phase,
    "bottom_left_second_phase": bottom_left_second_phase,
    "bottom_right_second_phase": bottom_right_second_phase,
    "top_right_second_phase": top_right_second_phase,
    "top_left_second_phase_inside": top_left_second_phase_inside,
    "bottom_left_second_phase_inside": bottom_left_second_phase_inside,
    "bottom_right_second_phase_inside": bottom_right_second_phase_inside,
    "top_right_second_phase_inside": top_right_second_phase_inside,
})

#%%

# df=df.iloc[:3975,:]
df["IoU"]=IoUs
df["recall"]=recalls
df["precisions"]=precisions


df.to_csv("results2.csv",index=False)