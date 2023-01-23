# food-classifier
A convolutional neural network that can identify different foods from an image. The neural network is trained on the 
[Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The Food101 dataset contains 101 different classes of food and a total of 101,000 images. There are 750 training images and 250 testing images per class,
with a total of 1000 images per class.

Currently, this repository only contains results for a model trained on 3 classes, but it will be expanded to 101 classes.

## Outline
This repository contains (or will contain) two sections:

1. [Experiment](#1---experiment)
    * [1.1 - Results](#i-loss--accuracy-curves)
      1. [Loss & Accuracy Curves](#i-loss--accuracy-curves)
      2. [Training and Test Loss & Accuracy Values](#ii-training-and-test-loss--accuracy-values)
      3. [Accuracy vs. Loss Plot](#iii-accuracy-vs-loss-plot)
      4. [Best Model](#iv-best-model)
2. [Final Model](#2---final-model)

## 1 - Experiment
This section contains the experimental part of the project. The experiment involves testing the convolutional neural network with various
hyperparameter combinations on a subset of the full dataset.

The subset of the dataset only contains 3 classes and 50% of the images per class. I.e., there are 375 training images and 125 testing images per class,
for a total of 1500 images.

There are a total of 8 different hyperparameter combinations that were tested. The hyperparameters that were varied were: the augmentation intensity, the
`p` value for the one dropout layer (i.e. how many neurons get dropped), and the learning rate (for the `torch.optim.Adam` optimizer).

## 1.1 - Results

### (i) Loss & Accuracy Curves

--         |  --
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/104711470/213962088-f50e7944-44ca-4d7c-9102-a693cec0c70f.png)  |  ![image](https://user-images.githubusercontent.com/104711470/213962160-8c35eb88-ead5-4a62-a2ce-a9ac444d3fc4.png)

**Figure 1.1: Loss and accuracy curves for each model.**

### (ii) Training and Test Loss & Accuracy Values
**Note:** The following values are averaged over the last 5 epochs to avoid issues from fluctuating loss or accuracy values

Model Name|train_loss|	train_acc|	test_loss|	test_acc|
:---:|:---:|:---:|:---:|:---:
model_0 |	0.52|	80.3%|	0.56|	76.6%
model_1	| 0.52|	79.8%|	0.61|	74.5%
model_2	| 0.58|	77.2%|	0.55|	78.4%
model_3	| 0.66|	73.0%|	0.62| 76.2%
model_4 |	0.76|	66.6%|	0.72|	69.2%
model_5	| 0.74|	68.0%|	0.71|	69.7%
model_6	| 0.74|	68.5%|	0.68|	72.2%
model_7	| 0.80|	64.8%|	0.71|	71.2%

**Table 1.1: Training and test loss and accuracy values averaged over the last 5 epochs.**

From table 1.1, we can see that models 0 to 3 have the highest test accuracies. 

There are some models that do better on the training data than they do on the testing data, which tells us that the model is overfitting there. 

There are also models that are doing better on the testing data than they are on the training data. This is likely due to the fact that those models use augmentation which is introducing a lot of errors into the training data labels but is at least preparing our model well for unseen results.

**TODO:** Look into this problem further to see if testing accuracy > training accuracy is really an issue.

### (iii) Accuracy vs. Loss Plot
For viewing the relation between training accuracy and loss or testing accuracy and loss, we look at the following plot:

![image](https://user-images.githubusercontent.com/104711470/213963071-bc8f9621-88b4-4bcb-9e01-faa64134e560.png)

**Figure 1.2: Accuracy vs loss plot.**

**Note:** The dashed lines in figure 1.2 are a bit arbitrary and only there to make it easier to separate points into high accuracy - loss loss, high accuracy - high loss, etc.

Generally, the closer the test and training points are, the better the model since the closeness of the points indicates whether how appropriately our model fits the data (or possibly underfits). Further away points means that the model is overfitting (i.e. it fits training data better than test data) or there's some other issue causing testing accuracy to be better than the training accuracy. The former is due to not having much variation in the dataset or learning on the training dataset too quickly which leads to the model learning the training data very well, but not performing so well on unseen (test) data.

Model Name | Distance
------------- | -------------
model_5 |	0.00127
model_2 |	0.00143
model_4 |	0.00250
model_3 |	0.00270
model_0 |	0.00367
model_6 |	0.00446
model_1 |	0.01226
model_7 |	0.01297

**Table 1.2: Distance between the points (training loss, training accuracy) and (testing loss, testing accuracy) listed in ascending order from top to bottom.**

Organizing the points in figure 1.2 gives us table 1.2. This shows us that models 3 to 5 have training and testing loss and accuracy values the closest to each other.

### (iv) Best Model
Based on the following criteria:
1. The best model must have one of the highest testing accuracies
2. The best model must also be fitting appropriately

It would appear that `model_2` is the best model. The hyperparameters for `model_2` are
* augmentation intensity = 0
* dropout value = 0.5
* learning rate = 0.001

We will try these hyperparameters out for our final model.

## 2 - Final Model

...Coming soon...