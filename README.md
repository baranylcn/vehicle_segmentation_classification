# Vehicle Segmentation and Classification
This project focuses on segmenting and classifying images of vehicles using a deep learning approach with TensorFlow and Keras. The dataset consists of various vehicle images, and the model is trained to classify these images into different categories.

# Project Overview
This project utilizes a convolutional neural network (CNN) built on a pre-trained MobileNetV3 model to classify vehicle images. The project includes data preprocessing, model training, and evaluation steps, and it visualizes the results of the model's performance.

# Dataset
The dataset used in this project consists of images of vehicles, organized into subfolders representing different classes. The images were collected and tagged from the internet and are loaded and processed using the ImageDataGenerator class from Keras.
![image](https://github.com/baranylcn/deneme/assets/98966968/ed224b42-6fd8-4f1e-b42d-62200f4e8b9f)


# Model Architecture
The model is built on the pre-trained MobileNetV3 architecture with additional dense layers for classification. The architecture includes:
- Input resizing and rescaling layers
- Data augmentation layers
- MobileNetV3 base model
- Additional dense layers with dropout for regularization
- Output layer with softmax activation
# Training Model
The model is trained with the following parameters:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall
- Epochs: 20
- Batch size: 32

# Results
The model's performance is evaluated on a separate test dataset. The results include test loss and accuracy, which are printed after training. Additionally, training and validation accuracy/loss plots are generated to visualize the model's performance over epochs.
- Test Accuracy: 91.54%
![image](https://github.com/baranylcn/deneme/assets/98966968/247a5951-c92f-420c-bdbc-396e080ee8b0)

