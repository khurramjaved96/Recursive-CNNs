import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import model
import evaluation

cornerModel_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\cornerModelPyTorch"

documentModel_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\model-data\DocumentModelPyTorch"

corner_refiner = evaluation.corner_refiner.corner_finder(cornerModel_path)

model = model.ModelFactory.get_model("resnet", 'document')
model_data_dict = torch.load(documentModel_path, map_location='cpu')
model_state_dict = model.state_dict()
missing_layers_keys = set([x for x in model_state_dict.keys()]) - set([x for x in model_data_dict.keys()])
missing_layers = {x: model_state_dict[x] for x in missing_layers_keys}
model_data_dict.update(missing_layers)
model.load_state_dict(model_data_dict)
# %%

pil_image = Image.open(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train\background01\datasheet002.avi\003.jpg")
# pil_image=Image.open(r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train\background01\tax005.avi\070.jpg")

# %%
pil_image = np.array(pil_image)
with torch.no_grad():
    image_array = np.copy(pil_image)
    pil_image = Image.fromarray(pil_image)
    test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                         transforms.ToTensor()])
    img_temp = test_transform(pil_image)

    img_temp = img_temp.unsqueeze(0)
    if torch.cuda.is_available():
        img_temp = img_temp.cuda()

    model_prediction = model(img_temp).cpu().data.numpy()[0]

model_prediction = np.array(model_prediction)

x_cords = model_prediction[[0, 2, 4, 6]]
y_cords = model_prediction[[1, 3, 5, 7]]

fig, ax = plt.subplots(2, 1)
resized = pil_image.resize((32, 32))
# plt.text
ax[0].imshow(pil_image)
ax[0].scatter(x_cords * image_array.shape[1], y_cords * image_array.shape[0])
for i in range(len(["tl", "tr", "br", "bl"])):
    print(i)
    ax[0].text(x_cords[i] * image_array.shape[1], y_cords[i] * image_array.shape[0], ["tl",  "bl", "br","tr"][i])

ax[1].imshow(resized)
ax[1].scatter(x_cords * 32, y_cords * 32)

for i in range(len(["tl", "tr", "br", "bl"])):
    ax[1].text(x_cords[i] * 32, y_cords[i] * 32, ["tl",  "bl", "br","tr"][i])
plt.show()

x_cords = x_cords * image_array.shape[1]
y_cords = y_cords * image_array.shape[0]
# %%


top_left_y_lower_bound = max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2))
top_left_y_upper_bound = int((y_cords[3] + y_cords[0]) / 2)
top_left_x_lower_bound = max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2))
top_left_x_upper_bound = int((x_cords[1] + x_cords[0]) / 2)

top_right_y_lower_bound = max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2))
top_right_y_upper_bound = int((y_cords[1] + y_cords[2]) / 2)
top_right_x_lower_bound = int((x_cords[1] + x_cords[0]) / 2)
top_right_x_upper_bound = min(image_array.shape[1] - 1, int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))

bottom_right_y_lower_bound = int((y_cords[1] + y_cords[2]) / 2)
bottom_right_y_upper_bound = min(image_array.shape[0] - 1, int(y_cords[2] + (y_cords[2] - y_cords[1]) / 2))
bottom_right_x_lower_bound = int((x_cords[2] + x_cords[3]) / 2)
bottom_right_x_upper_bound = min(image_array.shape[1] - 1, int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))

bottom_left_y_lower_bound = int((y_cords[0] + y_cords[3]) / 2)
bottom_left_y_upper_bound = min(image_array.shape[0] - 1, int(y_cords[3] + (y_cords[3] - y_cords[0]) / 2))
bottom_left_x_lower_bound = max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2))
bottom_left_x_upper_bound = int((x_cords[3] + x_cords[2]) / 2)


# %%

print(f"top_left_y_lower_bound = {top_left_y_lower_bound:.2f} ")
print(f"top_left_y_upper_bound = {top_left_y_upper_bound:.2f} ")
print(f"top_left_x_lower_bound = {top_left_x_lower_bound:.2f} ")
print(f"top_left_x_upper_bound = {top_left_x_upper_bound:.2f} ")

print(f"top_right_y_lower_bound = {top_right_y_lower_bound:.2f} ")
print(f"top_right_y_upper_bound = {top_right_y_upper_bound:.2f} ")
print(f"top_right_x_lower_bound = {top_right_x_lower_bound:.2f} ")
print(
    f"top_right_x_upper_bound = {top_right_x_upper_bound:.2f} ")

print(f"bottom_right_y_lower_bound = {bottom_right_y_lower_bound:.2f} ")
print(
    f"bottom_right_y_upper_bound = {bottom_right_y_upper_bound:.2f} ")
print(f"bottom_right_x_lower_bound = {bottom_right_x_lower_bound:.2f} ")
print(
    f"bottom_right_x_upper_bound = {bottom_right_x_upper_bound:.2f} ")

print(f"bottom_left_y_lower_bound = {bottom_left_y_lower_bound:.2f} ")
print(
    f"bottom_left_y_upper_bound = {bottom_left_y_upper_bound:.2f} ")
print(f"bottom_left_x_lower_bound = {bottom_left_x_lower_bound:.2f} ")
print(f"bottom_left_x_upper_bound = {bottom_left_x_upper_bound:.2f} ")
#%%

print(f"top_left = image_array[{top_left_y_lower_bound}:{top_left_y_upper_bound},\n           {top_left_x_lower_bound}:{top_left_x_upper_bound}]\n")

print(f"top_right = image_array[{top_right_y_lower_bound}:{top_right_y_upper_bound},\n            {top_right_x_lower_bound}:{top_right_x_upper_bound}]\n")

print(f"bottom_right = image_array[{bottom_right_y_lower_bound}:{bottom_right_y_upper_bound},\n               {bottom_right_x_lower_bound}:{bottom_right_x_upper_bound}]\n")

print(f"bottom_left = image_array[{bottom_left_y_lower_bound}:{bottom_left_y_upper_bound},\n              {bottom_left_x_lower_bound}:{bottom_left_x_upper_bound}]\n")
# %%
top_left = image_array[top_left_y_lower_bound:top_left_y_upper_bound,
           top_left_x_lower_bound:top_left_x_upper_bound]

top_right = image_array[top_right_y_lower_bound:top_right_y_upper_bound,
            top_right_x_lower_bound:top_right_x_upper_bound]

bottom_right = image_array[bottom_right_y_lower_bound:bottom_right_y_upper_bound,
               bottom_right_x_lower_bound:bottom_right_x_upper_bound]

bottom_left = image_array[bottom_left_y_lower_bound:bottom_left_y_upper_bound,
              bottom_left_x_lower_bound:bottom_left_x_upper_bound]

top_left = (top_left,
            top_left_x_lower_bound,
            top_left_y_lower_bound,
            top_left_x_upper_bound,
            top_left_y_upper_bound)

top_right = (top_right,
             top_right_x_lower_bound,
             top_right_y_lower_bound,
             top_right_x_upper_bound,
             top_right_y_upper_bound)

bottom_right = (bottom_right,
                bottom_right_x_lower_bound,
                bottom_right_y_lower_bound,
                bottom_right_x_upper_bound,
                bottom_right_y_upper_bound)

bottom_left = (bottom_left,
               bottom_left_x_lower_bound,
               bottom_left_y_lower_bound,
               bottom_left_x_upper_bound,
               bottom_left_y_upper_bound)
