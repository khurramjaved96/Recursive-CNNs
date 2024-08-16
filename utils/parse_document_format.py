import os
import pandas as pd
import xml.etree.ElementTree as ET

images_path = r"/datasets/testDataset/smart-doc-train\background01"
r'C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\background01\datasheet002.gt.xml'


def xml_to_df(xml_file, images_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    x_shape = 1920
    y_shape = 1080
    segmentations_results = root[2]
    paths = []
    tls = []
    trs = []
    brs = []
    bls = []
    for frame in segmentations_results:
        base_index = frame.attrib["index"]
        base_name = f"{int(base_index):03d}.jpg"
        path_name = os.path.join(images_path, base_name)
        paths.append(path_name)
        corners = frame.findall("point")
        corners = {corner.attrib["name"]: [float(corner.attrib["x"]) / x_shape, float(corner.attrib["y"]) / y_shape] for
                   corner in corners}
        tl = corners["tl"]
        tr = corners["tr"]
        br = corners["br"]
        bl = corners["bl"]

        tls.append(tl)
        trs.append(tr)
        brs.append(br)
        bls.append(bl)
    df = pd.DataFrame({"path": paths, "tl": tls, "tr": trs, "br": brs, "bl": bls})
    return df


base_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset"

# images_base_path=r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\documents"

backgrounds = ["background01",
               "background02",
               "background03",
               "background04",
               "background05"]

df_lists = []
for bg in backgrounds:
    bg_path = os.path.join(base_path, bg)
    for file in os.listdir(bg_path):
        if file.endswith(".xml"):
            xml_path = os.path.join(bg_path, file)
            file_name = file.replace(".gt.xml", ".avi")
            imgs_path = os.path.join(bg, file_name)
            entry_df = xml_to_df(xml_path, imgs_path)
            df_lists.append(entry_df)
            # break

    # break
# %%
complete_df = pd.concat(df_lists)
all(entry_df["path"].apply(lambda x: os.path.exists(x)).tolist())
complete_df.to_csv(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\documents\gt1.csv",
    index=False,
    columns=None)
# %%
import csv

input_file = r'/datasets/testDataset/smart-doc-train\gt1.csv'
output_file = r'/datasets/testDataset/smart-doc-train\gt.csv'

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
