# Experiment 1
For this section, we look at only 3 classes. That means we have 3000 images total, with 2250 train images and 750 test images.

## Outline
1. [**Introduction**](README.md#introduction)
2. [**Experiment Results**](README.md#experiment-results)
   1. [Loss & Accuracy Curves](#i-loss--accuracy-curves)
   2. [Training and Test Loss & Accuracy Values](#ii-training-and-test-loss--accuracy-values)
   3. [Accuracy vs. Loss Plot](#iii-accuracy-vs-loss-plot)
   4. [Best Model](#iv-best-model)
3. [**Experiment's Final Model Results**](README.md#experiments-final-model-results)
   1. [Training and Test Loss & Accuracy Values](#i-training-and-test-loss--accuracy-values)
   2. [Loss & Accuracy Curves](README.md#ii-loss--accuracy-curves)
4. [**False Ending, More Experimenting**](README.md#false-ending-more-experimenting)
   1. [Training and Test Loss & Accuracy Values](README.md#i-training-and-test-loss--accuracy-values-1)
   2. [Loss & Accuracy Curves](README.md#ii-loss--accuracy-curves-1)
   3. [Predictions on Test Data Visualized](README.md#iii-predictions-on-test-data-visualized)
   4. [ Confusion Matrix](README.md#iv-confusion-matrix)

## Introduction
This section contains the first experimental part of the project. The first experiment involves testing the convolutional neural network with various
hyperparameter combinations on a subset of the full dataset.

The subset of the dataset only contains 3 classes and 50% of the images per class. I.e., there are 375 training images and 125 testing images per class,
for a total of 1500 images.

There are a total of 8 different hyperparameter combinations that were tested. The hyperparameters that were varied were: the augmentation intensity, the
`p` value for the one dropout layer (i.e. how many neurons get dropped), and the learning rate (for the `torch.optim.Adam` optimizer).

The following model architecture was used:

![CNN Arch - Exp 1 - 2b 2l labelled](https://user-images.githubusercontent.com/104711470/216860485-1e57270c-fac1-4ad7-8b61-659fd5c71d12.png)

**Figure 0.1:** CNN model architecture for experiment 1. Uses `kernel_size = 3`, `stride = 1`, and `padding = 0` for the Conv2d layers, and `kernel_size = 2` for the MaxPool2d layers.

### Experiment Results

**Note:** There was a bug in the code where I set it up so that the testing accuracies were evaluated using an augmented test dataset. It shouldn't be too much of an issue since it just means that the model does well on a variety of different images.

#### (i) Loss & Accuracy Curves

--         |  --
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/104711470/213962088-f50e7944-44ca-4d7c-9102-a693cec0c70f.png)  |  ![image](https://user-images.githubusercontent.com/104711470/213962160-8c35eb88-ead5-4a62-a2ce-a9ac444d3fc4.png)

**Figure 1:** Loss and accuracy curves for each model.

#### (ii) Training and Test Loss & Accuracy Values
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

**Table 1:** Training and test loss and accuracy values averaged over the last 5 epochs.

From table 1, we can see that models 0 to 3 have the highest test accuracies. 

There are some models that do better on the training data than they do on the testing data, which tells us that the model is overfitting there. 

There are also models that are doing better on the testing data than they are on the training data. This is likely due to the fact that those models use augmentation which is introducing a lot of errors into the training data labels but is at least preparing our model well for unseen results.

**TODO:** Look into this problem further to see if testing accuracy > training accuracy is really an issue.

**Explanation:** It likely isn't a bad thing at all. The training accuracy was evaluated without `torch.inference_mode()` meaning that the dropout layer was active when calculating these values. The testing accuracy was evaluated using `torch.inference_mode()`, i.e. with no neurons being dropped, so it will naturally be higher since there's now more information available to make a decision with. There's also the part about augmnetation mentioned above.

#### (iii) Accuracy vs. Loss Plot
For viewing the relation between training accuracy and loss or testing accuracy and loss, we look at the following plot:

![image](https://user-images.githubusercontent.com/104711470/213963071-bc8f9621-88b4-4bcb-9e01-faa64134e560.png)

**Figure 2:** Accuracy vs loss plot.

**Note:** The dashed lines in figure 2 are a bit arbitrary and only there to make it easier to separate points into high accuracy - loss loss, high accuracy - high loss, etc. Also note that the values that are plotted here are from table 1.

Generally, the closer the test and training points are, the better the model since the closeness of the points indicates how appropriately our model fits the data (or possibly underfits). Further away points means that the model is overfitting (i.e. it fits training data better than test data) or there's some other issue causing testing accuracy to be better than the training accuracy. The former is due to not having much variation in the dataset or learning on the training dataset too quickly which leads to the model learning the training data very well, but not performing so well on unseen (test) data. The latter can be caused by underfitting, but is usually more likely due to dropout layers causing the training accuracy to be reduced since the training dataset has fewer neurons (and thus less information) to make a decision with or augmentation which is introducing some "errors" in the training dataset, but which prepares the model well for unseen data (the testing dataset).

Model Name | Distance
------------- | -------------
model_5 | 0.0356
model_2 | 0.0378
model_4 | 0.0500
model_3 | 0.0520
model_0 | 0.0606
model_6 | 0.0668
model_1 | 0.111
model_7 | 0.114

**Table 2:** Distance between the points (training loss, training accuracy) and (testing loss, testing accuracy) listed in ascending order from top to bottom.

The distance was calculated using the standard euclidean distance formula:

$$\text{distance}_i = \sqrt{(\text{training accuracy}_i - \text{test accuracy}_i)^2 + (\text{training loss}_i - \text{test loss}_i)^2}$$

where $i$ is the number of the model.

Organizing the points in figure 2 gives us table 2. This shows us that models 2 to 5 have training and testing loss and accuracy values the closest to each other.

#### (iv) Best Model
Based on the following criteria:
1. The best model must have one of the highest testing accuracies
2. The best model must also be fitting appropriately (in our case, anything with testing accuracy $>\approx$ training accuracy  is good)

It would appear that `model_2` is the best model. The hyperparameters for `model_2` are
* augmentation intensity = 0
* dropout value = 0.5
* learning rate = 0.001

We will try these hyperparameters out for our final model.

### Experiment's Final Model Results

Here, I used the best model's hyperparameters to train the model again but this time, on a dataset containing all 1000 images per class, with 750 images for training and 250 for testing. Also note that the model was trained for 40 epochs. Total training time was 7.8 min (training device was set to "cuda").

Just to reiterate, the hyperparameters for `model_2`, and thus also for the final model, are
* augmentation intensity = 0
* dropout value = 0.5
* learning rate = 0.001

#### (i) Training and Test Loss & Accuracy Values

train_loss | train_acc |test_loss | test_acc    
-----| -------- | ------| -------
0.46 | 81.9% |  0.46 |  81.2%

**Table 3:** Training and test accuracy and loss values for the final experimental model.

It appears that training on 1500 more images and 10 more epochs than before gave only a very marginal improvement.

**TODO:** Explore why that is.

#### (ii) Loss & Accuracy Curves

![image](https://user-images.githubusercontent.com/104711470/214213221-7c580b7f-dcfe-44d0-a0b7-822796329a9d.png)

**Figure 3:** Loss and accuracy curves for final experimental model.

From figure 3, we can see that accuracies don't increase much between epochs 30 and 40. 

**TODO:** Need to look into why that is. This does explain why changing the epochs didn't change accuracy by much, but need to look into why doubling the dataset did nothing.

### False Ending, More Experimenting
After seeing the previous section's model not improve much in accuracy from before, I tried different hyperparameters again. I used the same dataset as in
section 1.2, but I changed the following hyperparameters:
* `hidden_units`: from 10 in section 1.1 -> 20 now
* `learning_rate`: from 0.001 -> 0.002
* `dropout_value`: from 0.5 -> 0.75
The other hyperparameters remained the same.

#### (i) Training and Test Loss & Accuracy Values

train_loss|	train_acc|	test_loss|	test_acc
---|---|---|---
0.44	|83.0%|	0.36|	85.5%

**Table 4:** Training and test accuracy and loss values for the final experimental model adjusted version.

The improvement from the final model to the adjusted final model is 4.3%. Marginal, but still indicates that our change in hyperparameters helped. The higher learning rate and increased number of hidden units would lead to faster learning, but the increased dropout value makes it so that it's not so fast as to overfit. Getting lucky with the trade-off between the three is likely what lead to the marginal improvement.

#### (ii) Loss & Accuracy Curves

![image](https://user-images.githubusercontent.com/104711470/214232135-7b89d81c-dd82-42f4-b433-62a9195af2eb.png)

**Figure 4:** Loss and accuracy curves for final experimental model adjusted version.

The significant difference between the training and testing accuracies in figure 4 is due to the large dropout value. The training accuracies were calculated with the dropout layer activated, while the testing accuracies were calculated with the dropout layer turned off. Since our model learned well, it will naturally end up doing worse with the dropout layer on.


#### (iii) Predictions on Test Data Visualized

Here we look at what the model predicted for 10 different test images from each class. The correct prediction will have a green title and the wrong predictions will have a red title.

![image](https://user-images.githubusercontent.com/104711470/214233451-00fe0df4-6d0e-4d03-998c-30bde5fafa6b.png)

**Figure 5:** 10 test images for 'beef_carpaccio' class and their class as predicted by the adjusted final experimental model.


![image](https://user-images.githubusercontent.com/104711470/214233921-075d9d33-379b-43a9-a4f9-6511e5931232.png)

**Figure 6:** 10 test images for 'carrot_cake' class and their class as predicted by the adjusted final experimental model.


![image](https://user-images.githubusercontent.com/104711470/214234192-82692aa2-32a7-4c59-b791-b9bbe4c8369a.png)

**Figure 7:** 10 test images for 'ramen' class and their class as predicted by the adjusted final experimental model.

It appears from figures 1.5-1.7 that the classes have the following accuracies:
* 'beef_carpaccio' accuracy = 9/10
* 'carrot_cake' accuracy = 7/10
* 'ramen' accuracy = 8/10

It seems that the model learned the 'beef_carpaccio' class very well, and didn't learn 'carrot_cake' very well. Let's take a look at the confusion matrix.

#### (iv) Confusion Matrix

<p align="center">
  <img src="https://user-images.githubusercontent.com/104711470/214237631-b50242b4-cd15-488a-9c6f-be25ad3f44eb.png" width = 400px/>
</p>

**Figure 8:** Confusion matrix of adjusted final experimental model.

This tells us that the model learned 'beef_carpaccio' very well, but often tends to confuse other food items for `ramen`. This is likely due to the fact that beef carpaccio has a different colour from ramen and carrot cake, which are more similar in colour. Ramen also consists of many different food items with a variety of shapes, so it is more likely to get confused with other food items. 

Since there are 250 test images per class, this matrix also gives us the accuracies per class. They are summarized in table 5 below.

Class Name | Accuracy
---|---
beef_carpaccio | 90.0%
carrot_cake | 83.2%
ramen| 83.2%

**Table 5:** Accuracies per class for adjusted final experimental model.
