import dataprocessor

import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
data_dirs = [
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations",
    # r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc"
]
dataset_type = "complete_document"


dataset = dataprocessor.DatasetFactory.get_dataset(data_dirs, dataset_type, "train.csv")

train_dataset_loader = dataprocessor.LoaderFactory.get_loader("hdd_complete_doc", dataset.myData,
                                                              transform=None,
                                                              cuda=False)

#%%
corner_1=[]
corner_2=[]
corner_3=[]
full=[]

for idx in range(len(train_dataset_loader) ):
    sample_corner = train_dataset_loader[idx]
    if sample_corner[2]=="complete_doc":
        full.append(sample_corner)
    elif sample_corner[2]=="1_corners":
        corner_1.append(sample_corner)
    elif sample_corner[2]=="2_corners":
        corner_2.append(sample_corner)
    elif sample_corner[2]=="3_corners":
        corner_3.append(sample_corner)

#%%

index=random.randint(0,len(corner_3))
sample_corner=corner_3[index]

img = np.array(sample_corner[0])

fig, ax = plt.subplots(2,2)

# img=np.transpose(img, (1, 2, 0))
img_label = np.array(img)
ax[0,0].imshow(img)
ax[0,0].set_title(str(sample_corner[2]))
# plt.show()

resize_size=64
resized=Image.fromarray(img).resize((resize_size,resize_size))
ax[0,1].imshow(resized)
ax[0,1].set_title("Resized "+str(resize_size))

resize_size=32
resized=Image.fromarray(img).resize((resize_size,resize_size))
ax[1,0].imshow(resized)
ax[1,0].set_title("Resized "+str(resize_size))
# plt.show()

resize_size=48
resized=Image.fromarray(img).resize((resize_size,resize_size))
ax[1,1].imshow(resized)
ax[1,1].set_title("Resized "+str(resize_size))
plt.show()


