import dataprocessor
import random
import random
import numpy as np

import matplotlib.pyplot as plt



# dataset_corner_collected  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\corner-datasets\kosmos-test"], "corner")
# loader_corner_collected=dataprocessor.LoaderFactory.get_loader("hdd", dataset_corner_collected.myData,
#                                                               transform=None,
#                                                               cuda=False)
#
#
# dataset_corner_collected  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\corner-datasets\smart-doc-train"], "corner")
# loader_corner_collected=dataprocessor.LoaderFactory.get_loader("hdd", dataset_corner_collected.myData,
#                                                               transform=None,
#                                                               cuda=False)


dataset_corner_collected  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\danie\OneDrive\Desktop\Trabajo Kosmos\Recursive-CNNs\corner-datasets\augmentations-train"], "corner")
loader_corner_collected=dataprocessor.LoaderFactory.get_loader("hdd", dataset_corner_collected.myData,
                                                              transform=None,
                                                              cuda=False)


#%%
index=random.randint(0,len(loader_corner_collected))
sample_corner=loader_corner_collected[index]
img = np.array(sample_corner[0])
# img=np.transpose(img, (1, 2, 0))
img_label = np.array(sample_corner[1])
x_cords = img_label[0 ]* img.shape[0]
y_cords = img_label[1 ]* img.shape[0]

fig, ax = plt.subplots()

ax.imshow(img)
ax.scatter(x_cords, y_cords)
# for index in range(len(x_cords)):
ax.text(x_cords, y_cords, ["tl"])
plt.show()
