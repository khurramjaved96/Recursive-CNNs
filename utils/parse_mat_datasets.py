import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


images_mat = scipy.io.loadmat(
    r'C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations\dtd_white_indoor_6600artimg.mat')
labels_mat = scipy.io.loadmat(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations\dtd_white_indoor_6600corners.mat")

sample_img = images_mat['allartimgs1'][0]
sample_label = labels_mat['allcorners1'][0]


fig, ax = plt.subplots()
ax.imshow(sample_img)
ax.scatter(sample_label[0], sample_label[1])
ax.text(sample_label[0], sample_label[1], "tl")
ax.scatter(sample_label[2], sample_label[3])
ax.text(sample_label[2], sample_label[3], "tr")
ax.scatter(sample_label[4], sample_label[5])
ax.text(sample_label[4], sample_label[5], "br")
ax.scatter(sample_label[6], sample_label[7])
ax.text(sample_label[6], sample_label[7], "bl")

plt.show()
#%%
paths = []
tls = []
trs = []
brs = []
bls = []
for idx in range(len(images_mat['allartimgs1'])):
    base_path=f"{int(idx):03d}.jpg"
    path=os.path.join(r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations",base_path)
    sample_img = images_mat['allartimgs1'][idx]
    Image.fromarray(sample_img).save(path)
    sample_label = labels_mat['allcorners1'][idx]
    tl, tr, br, bl = ([sample_label[0]/256, sample_label[1]/384],
                      [sample_label[2]/256, sample_label[3]/384],
                      [sample_label[4]/256,sample_label[5]/384],
                      [sample_label[6]/256, sample_label[7]/384])
    paths.append(base_path)
    tls.append(tl)
    trs.append(tr)
    brs.append(br)
    bls.append(bl)
df = pd.DataFrame({"path": paths, "tl": tls, "tr": trs, "br": brs, "bl": bls})
df.to_csv(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations\gt1.csv",
    index=False,
    columns=None)

import csv

input_file = r'C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations\gt1.csv'
output_file = r'C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations\gt.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in reader:
        new_row = []
        new_row.append(row[0])  # The file path remains unchanged
        coordinates = []

        for coordinate in row[1:]:
            coordinate = coordinate.strip('"')  # Remove surrounding quotes
            coordinate = coordinate.strip("[]")  # Remove surrounding brackets
            coordinates.append(f"[{coordinate}]")  # Add brackets around each coordinate

        # Join the coordinates with commas and surround with "|(|...|)|"
        new_coordinates = "|(" + ",".join(coordinates) + ")|"
        new_row.append(new_coordinates)

        writer.writerow(new_row)
