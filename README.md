# ECG-challenge
This repository presents an example of exploring bias in a 
neural network-based model predicting myocardial infarction from electrocardiograms (ECGs).

The work has been carried out in connection with the Data Science Spring School & Challenge for Early Career Researchers & Professionals 2022.

# Background
Electrocardiographic characteristics have been shown to be influenced by patient body mass index. 
# Aim
In this project, we aimed to explore if patient body mass index influences the performance of a neural network-based model predicting myocardial infarction from 12-lead ECGs.

# Experiment
see analysis.ipynb and prediction.ipynb.

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
resnet_18 with the first layer changed to accept a single channel 2D input.
```
    model = torchvision.models.resnet18(num_classes=1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2,bias=False)
```
Difference between conv1d and conv2d. 
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/conv1d.gif)  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/conv2d.gif)  
There is information exchange between different leads by con2d.
# Resutls  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/roc_curve.png)
## bias between different groups

Results from 10 experiments.  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/group_results.png)

Confusion matrix from one of the model by different groups.  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu_1.png)  
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu2.png)  
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu_3.png)



# Goal
# To do List
put in what we want to see from data

