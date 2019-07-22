# Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow

First step towards building an efficient OCR system is to find out the specific text locations. Implemented the YOLO ( You Only Look Once ) algorithm from scratch (no object detection API used) for the specific task of Scene Text Detection in python using keras and tensorflow.

## Data : 

The dataset used is ICDAR competetion dataset available here : [Drive Link](https://drive.google.com/open?id=1ObrV9pbH_-LBGbIodWgB6W4dtQloTTH6)
<br />
Train images = 376 <br />
Validation images = 115

## Preprocessing :

The `Preprocess.py` file handles all the necessary preprocessing and saves the data in the form of numpy arrays. First, the images are resized to (512,512) dimensions. Accordingly, the ground truth of the boxes is modified as well. All the images are normalized to a range of [-1 , 1]. The ground truth coordinates are processed to form a matrix of dimensions as ( grid height , grid width , 1 , 5 ). 

### Custom data :
Necessary changes need to be done in the `Preprocess.py` file to input the custom data and images, to create the appropriate input and output matrices. 

## Model :

MobileNetv2 architecture is used as a feture extractor. The main reason for choosing MobileNetv2 is the high accuracy and the less number of weights. The fully connected layers of MobileNet are removed. Three Conv layers are added to the last layer of the MobileNet architecture to output a shape of (grid height , grid width , 1 , 5 ). The model weights can be found here : [Drive Link](https://drive.google.com/open?id=1OwrEu6SeaNM3l_clLN9F40W-tMpRfz97)


## Loss Function and Training:

The loss function implemented is the one specified in the YOLO paper. As there is only one class to be predicted, the contribution of class predictions to the loss is eliminated. 
<br />

The model is trained for 180 epochs in total with a batch size of 4.  The learning rate was kept at 0.001 for the first 100 epochs and lowered to 0.0001 for the next 80 epochs. 

## Inference :

The `Utils.py` consists of the functions used to convert the matrix output ( grid height , grid width , 1 , 5 ) of the model to actual predicted bounding boxes. Non max suppression is used to eliminate boxes on the same object as stated in the YOLO paper.

## Results :

![alt text](https://github.com/Neerajj9/Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow/blob/master/Results1/28.jpg)
<br />
![alt text](https://github.com/Neerajj9/Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow/blob/master/Results1/113.jpg)
<br />
![alt text](https://github.com/Neerajj9/Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow/blob/master/Results1/114.jpg)
<br />

## Requirements : 

1. Keras : 2.2.2 
2. Tensorflow : 1.9.0 
3. OpenCV : 3.4.1 
4. Numpy : 1.14.3 
5. Matplotlib : 2.2.2 
