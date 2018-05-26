## Khurram Javed, Faisal Shaifait "Real-time Document Localization in Natural Images by Recursive Application of a CNN" 

![alt text](https://khurramjaved96.github.io/random/recursiveCNN.png "Logo Title Text 1")

Paper available at : https://khurramjaved96.github.io

## Demo
To test the system, you can use the pretrained models by:

``` bash
usage: python detectDocument.py [-i IMAGEPATH] [-o OUTPUTPATH]
                         [-rf RETAINFACTOR] [-cm CORNERMODEL]
                         [-dm DOCUMENTMODEL]
```
For example:
``` bash
python detectDocument.py -i TrainedModel/img.jpg -o ./result.jpg -rf 0.85
```
would run the pretrained model on the sample image in the repository. 

## Training Code
Training code is mostly for reference only. It's not well documented or commented and it would be easier to re-implement the model from the paper than using this code. However I will be refactoring the code in the coming days to make it more accesible. 

To prepare dataset for training, do the following: 
1. 
``` bash
python video_to_image --d ../path_to_smartdoc_videos/ --o ../path_to_store_frames
```
2. After converting to frames, we need to do random crops to assure variation in the data. This can be done by:
``` bash
python DocumentDataGenerator --d ../path_to_store_frames/ --o ../path_to_train_set
```
and
``` bash
python CornerDataGenerator --d ../path_to_store_frames/ --o ../path_to_train_set
```

Self-collected dataset can be downloaded from : https://drive.google.com/drive/folders/0B9Sr0v9WkqCmekhjTTY2aV9hUmM?usp=sharing

Email : 14besekjaved@seecs.edu.pk in-case of any queries. 


## Dataset format for training
For training, the dataset should be in following format:
1. .npy dumps of images. An example shape can be 10000x32x32x3. There should be two files; 1 for training and one for validation.
2. .npy dumps for ground truth. For document detector, shape should be 10000x8. For corner detector, it should be 100000x2.

## Note
To those working on this problem, I would encourage trying out fully connected neural networks (Or some variant of pixel level segmentation network) as well; in my limited experiments, they are able to out-perform my method quite easily, and are more robust to unseen backgrounds (Probably because they are able to utilize context information of the whole page when making the prediction). They do tend to be a bit slower and require more memory though (Because a high-res image is used as input.) 

