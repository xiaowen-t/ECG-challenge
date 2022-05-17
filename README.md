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

We split the train-test dataset according to the BMI.
## Testing dataset
For experiment 1, the testing dataset is devided into 3 groups according to WHO BMI grouping, each goup contains 128 samples:  
For experiment 2, the testing dataset is devided into 3 groups according to WHO BMI grouping, each goup contains 64 samples:  

Group 1:  
    Below 18.5 = Underweight  
    18.5–24.9 = Normal weight    

Group 2:  
    25.0–29.9 = Pre-obesity    

Group 3:  
    30.0–34.9 = Obesity class I  
    35.0–39.9 = Obesity class II    
    Above 40 = Obesity class III
## Training dataset
For eperiment 1, the training dataset contains all the other patients with MI annotation.  
For eperiment 2, the training dataset contains 320 patients from each group.
# Models
We simply adapt the `resnet_18` from Pytorch's official implementation. The first layer is changed to accept a single channel 2D input. (instead of 3 channel image data.)
```
    model = torchvision.models.resnet18(num_classes=1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2,bias=False)
```
Common way to use convolutional neural networks for ECG training is to use 1D convolution kernels. The difference between `conv1d` layer and `conv2d` layer is visualized below, conv2d layer is able to extract the internal and inter-lead features of the 12-lead ECG.

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/conv1d.gif)  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/conv2d.gif)  

We also used this model to predict the gender, notebook available on [Kaggle](https://www.kaggle.com/code/meansnothing/simple-binary-classification-with-resnet?scriptVersionId=94573461). Preliminary experiment with `resnet_152` achieved auc=0.93, which indicates that processing 12-lead ECG data directly using 2D CNN is at least not a worse approach.  

Such a result brings up more questions to think about, for example:
When building models for datasets from different fields, what are the focus points we need to consider?  
-> Whether the exchange of information during forward computation (the receptive fields and the communication between channels) is in line with the opinions of professionals?    

# Resutls  
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/roc_curve.png)
## bias between different groups  
Experiment 1:  
With imbalance classes.  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/group_results.png)  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/distribution_bmi.png)  

Experiment 2:  
Balanced by subsampling, the training data contains equal number of samples from different groups.    

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/group_results_equal.png)  

Confusion matrix from one of the model by different groups.  

![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu_1.png)  
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu2.png)  
![image](https://github.com/meansnothing/ECG-challenge/blob/main/docs/confu_3.png)



# Goal
# To do List
put in what we want to see from data

# Ref

BMI - Nutritional status  

Below 18.5 = Underweight  
18.5–24.9 = Normal weight  
25.0–29.9 = Pre-obesity  
30.0–34.9 = Obesity class I  
35.0–39.9 = Obesity class II  
Above 40 = Obesity class III
