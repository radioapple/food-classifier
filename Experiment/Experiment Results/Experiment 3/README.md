# Experiment 3

In this experiment, we will try out different model architectures to see which improves accuracy on the [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/#:~:text=The%20Food%2D101%20Data%20Set).

## Outline

1. [**Introduction**](#introduction)
2. [**Experiment Set Up**](#experiment-set-up)
   1. [Dataset](#i-dataset)
   2. [Model Architectures](#ii-model-architectures)
   3. [Hyperparameters](#iii-hyperparameters)
   4. [Model Names](#iv-model-names)
3. [**Results**](#results)
   1. [Loss Curves](#i-loss-curves)
   2. [Accuracy vs. Loss](#ii-accuracy-vs-loss)
4. [**Discussion**](#discussion)

## Introduction

In experiment 1, we tried out different hyperparameter combinations on the 2 blocks, 2 convolutional layers per block architecture. Here, we will try out
different model architectures instead to see which improves performance.

## Experiment Set Up

### (i) Dataset

For this experiment, I used 500 images per class, for 3 classes from the [Food101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/#:~:text=The%20Food%2D101%20Data%20Set).
Only the beef carpaccio, carrot cake, and ramen classes were used. This was only for the experimental stage though. After picking the "best" model, I ran
the same model for 40 epochs instead and used the full 1000 images per class, for 3 classes dataset instead.

### (ii) Model Architectures

I chose `kernel_size = 7`, `stride = 1`,  and `padding = 0`for the `Conv2d` layers, and `kernel_size = 2` for the maxpool layers. Each model contains
a different number of convolutional layers and blocks.

Number of Layers | 2 Blocks       |  3 Blocks
:-------------------------:|:-------------------------:|:-------------------------:
2 Conv2d Layers | ![CNN Arch - 2b 2l labelled](https://user-images.githubusercontent.com/104711470/216852998-e04f197c-4835-454c-ac7a-b42dc09db5d1.png)|  ![CNN Arch - 3b 2l labelled](https://user-images.githubusercontent.com/104711470/216853027-f588069c-a36f-4cd6-ae13-230d66dd75e9.png)
3 Conv2d Layers |![CNN Arch - 2b 3l labelled](https://user-images.githubusercontent.com/104711470/216853019-d6d161ce-09d1-4251-a41a-8afbc0fc7826.png) | ![CNN Arch - 3b 3l labelled](https://user-images.githubusercontent.com/104711470/216853033-f8c0a89b-183c-48dc-b75f-85ec712177d9.png)

**Table 1:** Model architectures for testing. M_1, M_2, and M_final values are given in table 2.

**Note:** The Food101 dataset has images that have at least 1 dimension as 512 pixels, and the other varies. In the figures in table 1, it says the original image size
is 3 x 512 x N, but it could also be 3 x N x 512, where N is the varying dimension.


Number of Blocks, Number of Layers | M_1 | M_2 | M_final
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
2, 2 | 116 | N/A | 2704
2, 3 | 113 | N/A | 2209
3, 2 | 116 | 52 | 400
3, 3 | 113 | 47 | 196

**Table 2:** M_1, M_2, and M_final values for the figures shown in table 1.

### (iii) Hyperparameters

I used the following hyperparameter values for each of the models:

* `input_size` = 224
* `augmentation_intensity` = 0 (i.e. no augmentation)
* `batch_size` = 32
* `hidden_units` = 10 (i.e. number of neurons in the hidden layers)
* `kernel_size` = 2 (for MaxPool2d layer)
* `dropout_value` = 0.5
* `learning_rate` = 0.001
* `epochs` = 10 (I only ran the experiment for 10 epochs)
* `conv_kernel_size` = 7 (Conv2d layer kernel size)
* `conv_stride` = 1 (Conv2d layer stride)
* `padding` = 0 (Conv2d layer padding)

### (iv) Model Names
The model names are given below

Model Name | Blocks | Layers | Hyperparameters
:-----:|:----:|:---:|:--------------------:
`model_0`| 2 | 2 | Given in section (iii)
`model_1`| 2 | 3 | Given in section (iii)
`model_2`| 3 | 2 | Given in section (iii)
`model_3`| 3 | 3 | Given in section (iii)
`model_4`| 3 | 2 | Given in section (iii), but with `augmentation_intensity` = 20 instead
`final_model` | 3 | 2 | Given in section (iii), but with `epochs` = 40 instead. Also note that the full 1000 images per class, for 3 classes dataset was used here.

**Table 3:** Model names and their corresponding number of convolutional blocks and layers and their hyperparameters.

The first run of the experiment gave models 0 to 3. The second run gave `model_4`, where I tried to see if augmentation would improve performance. The third
and final run gave `final_model`, which was just the best model from models 0 to 4, but run for 40 epochs instead of 10.

## Results

### (i) Loss Curves
<p align="center">
  <img src="https://user-images.githubusercontent.com/104711470/216855636-7694a9d9-8361-4484-997d-85d3d6e0db88.png" width = 500px/>
</p>

**Figure 1:** The loss curves for models 0 to 4. The x-axis is epochs and the y-axis is loss (left) or accuracy (right).

<p align="center">
  <img src="https://user-images.githubusercontent.com/104711470/216855875-15373a96-6f61-42b3-9e17-fcbdda1c2df1.png" width = 500px/>
</p>

**Figure 2:** Loss curves for `final_model`. The x-axis is epochs and the y-axis is loss (left) or accuracy (right).

Model Name | train_loss |	train_acc |	test_loss |	test_acc
:--------:|:--------:|:---------:|:---------:|:---------:
`model_0` |	0.800 |	65.6% |	1.05 |	54.6%
`model_1` |	0.939 |	57.0% |	1.00 |	53.4%
`model_2` |	0.956 |	54.1% |	0.972 |	55.6%
`model_3` |	1.10 |	31.2% |	1.10 |	32.6%
`model_4` |	1.10 |	30.2% |	1.10 |	34.9%
`final_model` |	0.551 |	77.54% |	0.574 |	76.24%

**Table 4:** Final loss and accuracy values for all models. Note that for model's 0 to 4, these values are after the 10th epoch. For `final_model`, these are after the 40th epoch.

### (ii) Accuracy vs. Loss

![image](https://user-images.githubusercontent.com/104711470/216856170-2331efd5-f8a5-4390-8b35-a44dec38c005.png)

**Figure 3:** Accuracy vs loss plot for all the models. 

**Notes for figure 3:** 
* The loss and accuracy values used were averaged over the last 5 epochs to avoid any issues due to fluctations. 
* `model_3`'s test accuracy vs loss point overlaps with `model_4`'s test accuracy vs loss. 
* The dashed lines represent the threshold I choose for what would be considered a "good" accuracy or loss.

## Discussion

It seems that a CNN model with 3 blocks, and 2 layers per block with `dropout_value = 0.5` gives us the best performance over a 10 epoch range. However, in the full run with 40 epochs and on the full 1000 images per class for 3 classes dataset, the performance of the best model was worse than the 2 blocks, 2 layers model from experiment 1 by 5% (the one shown in [figure 1.3 in the experiment 1 report](https://github.com/radioapple/food-classifier/edit/main/README.md#ii-loss--accuracy-curves)). We also seem to have a similar almost plateau in performance as before. The number of trainable parameters in the experiment 1 model is 8,083, whereas `final_model` has 15,063 trainable parameters. This tells us that the increased complexity of `final_model` from the experiment 1 model is unnecessary.

**TODO:** Figure out what that implies about whether we need more data, need to change other hyperparameters, or other actions we need to take.
