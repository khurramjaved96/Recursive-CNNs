import pandas as pd


import os


data_dirs = [
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations",
    # r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\kosmos",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\self-collected",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc"
]


def print_data(path):

    test=pd.read_csv(os.path.join(path,"test.csv"))
    train=pd.read_csv(os.path.join(path,"train.csv"))
    val=pd.read_csv(os.path.join(path,"val.csv"))
    print("|||||||||||||||||||||||||||||||||||")
    print(os.path.basename(path))
    print("train:")
    print("1 esquinas visibles:",dict(train["directory"].value_counts())["1_corners"])
    print("2 esquinas visibles:",dict(train["directory"].value_counts())["2_corners"])
    print("3 esquinas visibles:",dict(train["directory"].value_counts())["3_corners"])
    print("4 esquinas visibles:",dict(train["directory"].value_counts())["complete_doc"])

    print("test:")
    print("1 esquinas visibles:",dict(test["directory"].value_counts())["1_corners"])
    print("2 esquinas visibles:",dict(test["directory"].value_counts())["2_corners"])
    print("3 esquinas visibles:",dict(test["directory"].value_counts())["3_corners"])
    print("4 esquinas visibles:",dict(test["directory"].value_counts())["complete_doc"])

    print("val:")
    print("1 esquinas visibles:",dict(val["directory"].value_counts())["1_corners"])
    print("2 esquinas visibles:",dict(val["directory"].value_counts())["2_corners"])
    print("3 esquinas visibles:",dict(val["directory"].value_counts())["3_corners"])
    print("4 esquinas visibles:",dict(val["directory"].value_counts())["complete_doc"])

    print("|||||||||||||||||||||||||||||||||||")

 for dir in data_dirs:
    print_data(dir)
#%%
