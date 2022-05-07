# ECG-challenge
This repository presents an example of exploring bias in a 
neural network-based model predicting myocardial infarction from electrocardiograms.

The work has been carried out in connection with the Data Science Spring School & Challenge for Early Career Researchers & Professionals 2022.

# Background
Electrocardiographic characteristics have been shown to be influenced by patient body mass index. 
# Aim

# Experiment
    see prediction.ipynb.

    The testing dataset contains 300 patients, 100 for each BMI group (random selected). 
    
    3 Groups are separated according to the BMI values:  
    
    WHO BMI grouping

    BMI - Nutritional status
    Below 18.5 = Underweight
    18.5–24.9 = Normal weight
    25.0–29.9 = Pre-obesity
    30.0–34.9 = Obesity class I
    35.0–39.9 = Obesity class II
    Above 40 = Obesity class III

    Group 1: 
        Below 18.5 = Underweight
        18.5–24.9 = Normal weight  
    Group 2:
        25.0–29.9 = Pre-obesity  
    Group 3:
        30.0–34.9 = Obesity class I
        35.0–39.9 = Obesity class II  

    The training dataset contains all the other patients with MI annotation.
# Models
resnet_18 with the first layer adpated to single channel 2D input.
# Resutls
## bias between different groups


# Goal
# To do List
put in what we want to see from data

