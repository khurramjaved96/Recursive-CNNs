''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import model


class GetCorners:
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        # dummy_input=torch.randn(1, 3, 32, 32)
        # torch.onnx.export(
        #     self.model,  # Model to export
        #     dummy_input,  # Dummy input tensor
        #     "model_doc.onnx",  # Output file name
        #     export_params=True,  # Store the trained parameter weights inside the model file
        #     opset_version=11,  # ONNX version to export to (choose a suitable opset version)
        #     do_constant_folding=True,  # Whether to execute constant folding for optimization
        #     input_names=['input'],  # Name for the input tensor
        #     output_names=['output'],  # Name for the output tensor
        #     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow variable batch size
        # )
        #

        model_data_dict=torch.load(checkpoint_dir, map_location='cpu')
        model_state_dict=self.model.state_dict()
        missing_layers_keys=set([x for x in model_state_dict.keys()])-set([x for x in model_data_dict.keys()])
        missing_layers= {x: model_state_dict[x] for x in missing_layers_keys}
        model_data_dict.update(missing_layers)
        self.model.load_state_dict(model_data_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def get(self, pil_image,details=False):
        with torch.no_grad():
            image_array = np.copy(pil_image)
            pil_image = Image.fromarray(pil_image)
            test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])
            img_temp = test_transform(pil_image)

            img_temp = img_temp.unsqueeze(0)
            if torch.cuda.is_available():
                img_temp = img_temp.cuda()

            model_prediction = self.model(img_temp).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction)

            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            x_cords = x_cords * image_array.shape[1]
            y_cords = y_cords * image_array.shape[0]

            # Extract the four corners of the image. Read "Region Extractor" in Section III of the paper for an explanation.


            if details:

                top_left_y_lower_bound=max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2))
                top_left_y_upper_bound=int((y_cords[3] + y_cords[0]) / 2)
                top_left_x_lower_bound=max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2))
                top_left_x_upper_bound=int((x_cords[1] + x_cords[0]) / 2)


                top_right_y_lower_bound=max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2))
                top_right_y_upper_bound=int((y_cords[1] + y_cords[2]) / 2)
                top_right_x_lower_bound=int((x_cords[1] + x_cords[0]) / 2)
                top_right_x_upper_bound=min(image_array.shape[1] - 1,int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))


                bottom_right_y_lower_bound=int((y_cords[1] + y_cords[2]) / 2)
                bottom_right_y_upper_bound=min(image_array.shape[0] - 1, int(y_cords[2] + (y_cords[2] - y_cords[1]) / 2))
                bottom_right_x_lower_bound=int((x_cords[2] + x_cords[3]) / 2)
                bottom_right_x_upper_bound=min(image_array.shape[1] - 1,int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))


                bottom_left_y_lower_bound=int((y_cords[0] + y_cords[3]) / 2)
                bottom_left_y_upper_bound=min(image_array.shape[0] - 1, int(y_cords[3] + (y_cords[3] - y_cords[0]) / 2))
                bottom_left_x_lower_bound=max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2))
                bottom_left_x_upper_bound=int((x_cords[3] + x_cords[2]) / 2)



                top_left = image_array[top_left_y_lower_bound:top_left_y_upper_bound,
                                    top_left_x_lower_bound:top_left_x_upper_bound]

                top_right = image_array[top_right_y_lower_bound:top_right_y_upper_bound,
                                        top_right_x_lower_bound:top_right_x_upper_bound]

                bottom_right = image_array[bottom_right_y_lower_bound:bottom_right_y_upper_bound,
                                        bottom_right_x_lower_bound:bottom_right_x_upper_bound]

                bottom_left = image_array[bottom_left_y_lower_bound:bottom_left_y_upper_bound,
                                        bottom_left_x_lower_bound:bottom_left_x_upper_bound]

                top_left=(top_left,
                          top_left_x_lower_bound,
                          top_left_y_lower_bound,
                          top_left_x_upper_bound,
                          top_left_y_upper_bound)

                top_right=(top_right,
                           top_right_x_lower_bound,
                           top_right_y_lower_bound,
                           top_right_x_upper_bound,
                           top_right_y_upper_bound)

                bottom_right=(bottom_right,
                              bottom_right_x_lower_bound,
                              bottom_right_y_lower_bound,
                              bottom_right_x_upper_bound,
                              bottom_right_y_upper_bound)

                bottom_left=(bottom_left,
                             bottom_left_x_lower_bound,
                             bottom_left_y_lower_bound,
                             bottom_left_x_upper_bound,
                             bottom_left_y_upper_bound)


                return top_left, top_right, bottom_right, bottom_left
            else:
                top_left = image_array[
                           max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)):int(
                               (y_cords[3] + y_cords[0]) / 2),
                           max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)):int(
                               (x_cords[1] + x_cords[0]) / 2)]

                top_right = image_array[
                            max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)):int(
                                (y_cords[1] + y_cords[2]) / 2),
                            int((x_cords[1] + x_cords[0]) / 2):min(image_array.shape[1] - 1,
                                                                   int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))]

                bottom_right = image_array[int((y_cords[1] + y_cords[2]) / 2):min(image_array.shape[0] - 1, int(
                    y_cords[2] + (y_cords[2] - y_cords[1]) / 2)),
                               int((x_cords[2] + x_cords[3]) / 2):min(image_array.shape[1] - 1,
                                                                      int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))]

                bottom_left = image_array[int((y_cords[0] + y_cords[3]) / 2):min(image_array.shape[0] - 1, int(
                    y_cords[3] + (y_cords[3] - y_cords[0]) / 2)),
                              max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)):int(
                                  (x_cords[3] + x_cords[2]) / 2)]

                top_left = (top_left, max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)),
                            max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)))
                top_right = (
                    top_right, int((x_cords[1] + x_cords[0]) / 2),
                    max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)))
                bottom_right = (bottom_right, int((x_cords[2] + x_cords[3]) / 2), int((y_cords[1] + y_cords[2]) / 2))
                bottom_left = (bottom_left, max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)),
                               int((y_cords[0] + y_cords[3]) / 2))

                return top_left, top_right, bottom_right, bottom_left
