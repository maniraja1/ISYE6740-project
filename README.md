# ISYE6740-project

# Project Description

Class 2 interproximal decay occurs where adjacent teeth touch and it is the only type of decay that a dentist cannot directly see in a patient’s mouth. Class 2 interproximal decay can only be diagnosed with an x-ray image1.
This project aims to solve this problem and validate if machine learning techniques can be used to eliminate the ambiguity and subjectiveness surrounding class 2 interproximal decay.

We have experminted with differet machine learning algorthm to compare the reults and identify which model works better in classifying dental Xrays to identify class decay.

# Data pre processing 
A typical bitewing dental x-ray image contains 8 teeth. In order to account for the relative shape of each tooth within the classification algorithm, the Tooth Fairies propose manually or programmatically investigating 3 different parsing methods. 

* The first parsing method involves parsing each individual tooth image, cropped at the jaw bone and providing a label for each individual tooth. 
* The second parsing method involves looking at the “kissing point” between adjacent teeth and providing the ML algorithm with a label for the left tooth and right tooth. 

Once the X-rays are cropped, the images are standardized and resized using tensor flow. Each X-ray is resized to a [500 X 500 X 3] matrix

# Model validation

We have experimented with different modeling techniques. 

* KNN (K nearest neighbor)
* SVM (Suport vector machine)
* CNN (Convolution neural network)

The goal of this project is to report metrics around classifying type 2 decays using different modeling techiniques


# Code

Here is how the code is organized
* ImageProcessing consists of code required to resize/standardize X Ray images. The images are processed and stored under processed/processed and processed/processed using this code
* KNN consists of code required to train and validate class 2 decays using K nearest neighbor
* SVM consists of code required to train and validate class 2 decays using support vector machines
* Processed2 consists of X Ray images which were processed to include individual tooth image from a bitewing X Ray
* Processed3 consists of X Ray images which were processed to include the “kissing point” between adjacent teeth from a bitewing X Ray

