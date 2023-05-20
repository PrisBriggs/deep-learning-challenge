# deep-learning-challenge

Georgia Tech Data Science and Analytics BootCamp - May 2023

Homework Module 21 - Neural Networks and Deep Learning - Deep Learning Challenge
By Priscila Menezes Briggs

## Overview 

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks,this code uses the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, a CSV file was provided, containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Methodology

The dataset was preprocessed in order to prepare it for the neural network model. Some of the preprocess steps taken were establishing the target variable and dropping its correspondent column from the dataset, as well as establishing the features under analysis. Additional categorical columns were dropped and the remaining ones were enconded with get_dummies() function. The dataset was then split between training and testing and the data was scaled to be ready to be used by the model.

A neural network was designed using KerasTuner model to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. 

After analyzing the accuracy, the KerasTuner model was optimized with two attempts to achieve a target predictive accuracy higher than 75%. 

## Results

* Data Preprocessing
    * The target variable for the model was the feature IS_SUCCESSFUL, which had value 1 for "yes" and 0 for "no". 
    * All the remaining features, except EIN and NAME, were used as features for the model. 
    * EIN and NAME were initially removed from the input data because they are neither targets nor features.

* Compiling, Training, and Evaluating the Model
    * The neural network model chosen had the following characteristics:
        * 80 neurons in the input layer, 40 neurons in the hidden layer and 1 neuron in the output layer. Total = 121 neurons, resulting in 6,801 parameters. The number of dimensions in the input layer was the same as the number of features in the training dataset.
        * 1 input layer, 1 hidden layer and 1 output layer.
        * activation functions: ReLU for input and hidden layer and sigmoid for output layer. The sigmoid function values are normalized to a probability between 0 and 1, which is ideal for a binary classification dataset. The rectified linear unit (ReLU) function is ideal for modeling positive, nonlinear input data for classification or regression. The ReLU function is always a good starting point, but not all data are positive, especially when normalized.
    * This first attempt returned 72.61% of accuracy, which was under the target predictive accuracy of 75% but not by much, considering 55.64% loss.   
    * To increase model's performance, two other attempts of optimization were made.


* Optimizing the Model
    * The second neural network model chosen had the following characteristics:
        * Differently from the first attempt, this time the feature "NAME" was not dropped and remained part of the training dataset. 
        * 80 neurons in the input layer, 40 neurons in the hidden layer and 1 neuron in the output layer. Total = 121 neurons, resulting in 10,801 parameters. The number of dimensions in the input layer was the same as the number of features in the training dataset.
        * 1 input layer, 1 hidden layer and 1 output layer.
        * activation functions: ReLU for input and hidden layer and sigmoid for output layer. 
    * This second attempt returned 75.76% accuracy, which was slightly above the target predictive accuracy, considering 49.17% loss. 

    * The third neural network model chosen had the following characteristics:
        * Differently from the first attempt,  the feature "NAME" was not dropped and remained part of the training dataset. 
        * 80 neurons in the input layer, 70 neurons in the hidden layers and 1 neuron in the output layer. Total = 151 neurons, resulting in 11,801 parameters. The number of dimensions in the input layer was the same as the number of features in the training dataset.
        * 1 input layer, 3 hidden layers and 1 output layer.
        * activation functions: ReLU for input layer, sigmoid for hidden layers and  for output layer. 
    * This third attempt returned 75.77% accuracy, which was slightly above the target predictive accuracy and the number achieved in the second model, considering 49.22% loss.  

## Summary
All models used KerasTuner as the base model and iterated through 100 epochs.
The first model, with 121 neurons, one hidden layer and using ReLU as the activation function in the input layer, performed below the target predictive accuracy, resulting in only 73%, but both optimized models exceeded the target value of 75% of accuracy.
The model with the best performance was the third model, which returned 75.77% accuracy with 49.21% loss, using 3 hidden layers with sigmoid as the activation function. This model had 151 neurons. 
Losses decreased from 55.64% to just over 49% after model optimization. 
Keeping the NAME column was one of the factors that made it possible to overcome the target predictive accuracy, along with the increase in neurons and hidden layers.

A suggestion of a different model that could be used to solve this classification model would be with Machine Learning - Logistic Regression Model.  
This model is used for binary classification problems (those where the data has only two classes) and takes a linear combination of the predictor variables to estimate the probability of the outcome being 0 or 1. Because the probability is calculated as a linear combination of the predictor variables, logistic regression models are relatively straightforward to interpret.

## Availability to the public
The files used in this challenge are available in the GitHub's repository on https://github.com/PrisBriggs/deep-learning-challenge.git.


## References:

The references used in this Challenge were the activities and lessons given in class, the tutoring classes, and the websites below. 

All webpages were visited in May/2023.

* https://www.statology.org/pandas-drop-multiple-columns/
* https://www.geeksforgeeks.org/how-to-count-distinct-values-of-a-pandas-dataframe-column/
* https://softhints.com/pandas-how-to-filter-results-of-value_counts/
* https://www.scaler.com/topics/find-index-of-element-in-list-python/
* https://www.mathworks.com/campaigns/offers/next/choosing-the-best-machine-learning-classification-model-and-avoiding-overfitting.html
    