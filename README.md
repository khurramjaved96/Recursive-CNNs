## Khurram Javed, Faisal Shaifait "Real-time Document Localization in Natural Images by Recursive Application of a CNN" 

![alt text](https://khurramjaved96.github.io/random/recursiveCNN.png "Logo Title Text 1")

Paper available at : https://khurramjaved96.github.io

- Install Tensorflow >= 1.8.0
``` bash
pip install -r requirements.txt
```
- Pre-requisite:  
    - Tensorflow >= 1.8.0
    - Python3.6  
    - Numpy  
    - SciPy  
    - Opencv 4.0.0.21 for Python


- clone the source code 
``` bash
git clone -b server_branch https://github.com/xiaoyubing/Recursive-CNNs.git
```

## Quick Start:  
To test the system, you can use the pretrained models by:

``` bash
usage: python detectDocument.py [-i IMAGEPATH] [-o OUTPUTPATH]
                         [-rf RETAINFACTOR] [-cm CORNERMODEL]
                         [-dm DOCUMENTMODEL]
```
For example:
``` bash
python detectDocument.py -i TrainedModel/img.jpg -o TrainedModel/result.jpg -rf 0.85
```
would run the pretrained model on the sample image in the repository. 

## Datasets 
SmartDoc Competition 2 dataset : https://sites.google.com/site/icdar15smartdoc/challenge-1/challenge1dataset
Self-collected dataset : https://drive.google.com/drive/folders/0B9Sr0v9WkqCmekhjTTY2aV9hUmM?usp=sharing

## Training Code
Training code is mostly for reference only. It's not well documented or commented and it would be easier to re-implement the model from the paper than using this code. However I will be refactoring the code in the coming days to make it more accesible. 

To prepare dataset for training, run the following command following: 

``` bash
python video_to_image.py --d ../path_to_smartdoc_videos/ --o ../path_to_store_frames
```
here video_to_image.py is in the SmartDocDataProcessor folder. 

After converting to videos to frames, we need to convert the data into format required to train the models. We have to train two models. One to detect the four document corners, and the other to detect the a corner point in an image. To prepare data for the first model, run:
``` bash
python DocumentDataGenerator --d ../path_to_store_frames/ --o ../path_to_train_set
```
and for the second model, run:

``` bash
python CornerDataGenerator --d ../path_to_store_frames/ --o ../path_to_corner_train_set
```

You can also download a version of this data in the right format from here:   
[Baidu pan](https://pan.baidu.com/s/1pSJDvhWczeNYrv6epbSQgA) Code:zex0  
[Google Drive](https://drive.google.com/drive/folders/1N9M8dHIMt6sQdoqZ8Y66EJVQSaBTq9cX?usp=sharing)

Now we can use the data to train our models. To train the document detector (The model that detects 4 corners), run:

``` bash
python documentDetectorTrainer.py --i path_to_train_set/
```
and to train the corner detector, run:

``` bash
python cornerTrainer.py --i path_to_corner_train_set/ --o path_to_checkpoints/
``` 

Email : 14besekjaved@seecs.edu.pk in-case of any queries. 

## Note
To those working on this problem, I would encourage trying out fully connected neural networks (Or some variant of pixel level segmentation network) as well; in my limited experiments, they are able to out-perform my method quite easily, and are more robust to unseen backgrounds (Probably because they are able to utilize context information of the whole page when making the prediction). They do tend to be a bit slower and require more memory though (Because a high-res image is used as input.) 

