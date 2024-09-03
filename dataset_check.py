
import dataprocessor



#%%
dataset_smart_doc_train= dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train"], "document")
loader_smart_doc_train =dataprocessor.LoaderFactory.get_loader("hdd", dataset_smart_doc_train.myData,
                                              transform=None,
                                              cuda=False)


dataset_smart_doc_test  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\smartDocData_DocTestC"], "document")
loader_smart_doc_test=dataprocessor.LoaderFactory.get_loader("hdd", dataset_smart_doc_test.myData,
                                                              transform=None,
                                                              cuda=False)

dataset_augmentation  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations"], "document")
loader_augmentation=dataprocessor.LoaderFactory.get_loader("hdd", dataset_augmentation.myData,
                                                              transform=None,
                                                              cuda=False)

dataset_collected  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\self-collected"], "document")
loader_collected=dataprocessor.LoaderFactory.get_loader("hdd", dataset_collected.myData,
                                                              transform=None,
                                                              cuda=False)

dataset_kosmos  = dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\kosmos-dataset"], "document")
loader_kosmos=dataprocessor.LoaderFactory.get_loader("hdd", dataset_kosmos.myData,
                                                              transform=None,
                                                              cuda=False)
#%%

#%%
def graph_samples(sample):
    img = np.array(sample[0])
    # img=np.transpose(img, (1, 2, 0))
    img_label = np.array(sample[1])
    x_cords = img_label[[0, 2, 4, 6]] * img.shape[1]
    y_cords = img_label[[1, 3, 5, 7]] * img.shape[0]


    fig, ax = plt.subplots()

    ax.imshow(img)
    ax.scatter(x_cords, y_cords)
    for index in range(len(x_cords)):
        ax.text(x_cords[index], y_cords[index],["tl","tr","br","bl"][index])
    plt.show()
#%%
import random
import numpy as np

import matplotlib.pyplot as plt
#%%

sample_smart_doc_train=loader_smart_doc_train[random.randint(0,len(loader_smart_doc_train))]
graph_samples(sample_smart_doc_train)
#%%
sample_smart_doc_test=loader_smart_doc_test[random.randint(0,len(loader_smart_doc_test))]
graph_samples(sample_smart_doc_test)
#%%

sample_augmentation=loader_augmentation[random.randint(0,len(loader_augmentation))]
graph_samples(sample_augmentation)
#%%
sample_collected=loader_collected[random.randint(0,len(loader_collected))]
graph_samples(sample_collected)

#%%
index=random.randint(0,len(loader_kosmos))
# index=0
print(dataset_kosmos.myData[0][index])
sample_kosmos=loader_kosmos[index]
graph_samples(sample_kosmos)
# print(np.array(sample_kosmos[1]).reshape((4,2)))
# #%%
# img = np.array(sample_kosmos[0])
# # img=np.transpose(img, (1, 2, 0))
# img_label = np.array(sample_kosmos[1])
# x_cords = img_label[[0, 2, 4, 6]] * img.shape[1]
# y_cords = img_label[[1, 3, 5, 7]] * img.shape[0]
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(img)
# ax.scatter(x_cords, y_cords)
# for index in range(len(x_cords)):
#     print(index)
#     ax.text(x_cords[index], y_cords[index],["tl","tr","br","bl"][index])
# plt.show()
# #%%
# #%%
# index=random.randint(0,len(loader_kosmos))
# index=3800

# #%%
# img = np.array(sample_kosmos[0])
# # img=np.transpose(img, (1, 2, 0))
# img_label = np.array(sample_kosmos[1])
# x_cords = (img_label[[0, 2, 4, 6]]) * img.shape[1]
# y_cords = (1-img_label[[1, 3, 5, 7]]) * img.shape[0]
#
# fig, ax = plt.subplots()
#
# ax.imshow(img)
# ax.scatter(x_cords, y_cords)
#
# for iidx in range(len(x_cords)):
#     ax.text(x_cords[iidx], y_cords[iidx], ["tl", "tr", "br", "bl"][iidx])
# plt.show()
#
# #%%
# img_label = np.array(sample_kosmos[1])
# y_cords = (img_label[[0, 2, 4, 6]]) * img.shape[0]
# x_cords = (img_label[[1, 3, 5, 7]]) * img.shape[1]
#
# fig, ax = plt.subplots()
#
# ax.imshow(img)
# ax.scatter(x_cords, y_cords)
#
# for idx in range(len(x_cords)):
#     ax.text(x_cords[idx], y_cords[idx], ["tl", "tr", "br", "bl"][idx])
# plt.show()
#
#
#
#
#
#
# #%%
#
