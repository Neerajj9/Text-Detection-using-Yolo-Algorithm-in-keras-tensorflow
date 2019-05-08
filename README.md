# Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow

First step towards building an efficient OCR system is to find out the specific text locations. Implemented the YOLO ( You Only Look Once ) algorithm from scratch (no object detection API used) for the specific task of Scene Text Detection in python using keras and tensorflow.

## Data : 

The dataset used is ICDAR competetion dataset available here : [Drive Link]


Train images = 376
Validation images = 115

## Preprocessing :

The Preprocess.py file handles all the necessary preprocessing and saves the data in the form of numpy arrays.
All the images are normalized to a range of [-1 , 1]. The ground truth coordinates are processed to form a matrix of dimensions as ( grid height , grid width , 1 , 5 ). 

### Custom data :
Necessary changes need to be done in the Preprocess.py file to input the custom data and images, to create the appropriate input and output numpy arrays.

## Model :

The fully connected layers of MobileNetv2 are removed and it is used as a feature extractor. Three Conv layers are added to the last layer of the MobileNet architecture to output a shape of (grid height , grid width , 1 , 5 ). The model weights can be found here : 


## Training :

The model is trained for 180 epochs in total with a batch size of 4.  The learning rate was kept at 0.001 for the first 100 epochs and lowered to 0.0001 for the next 80 epochs.
