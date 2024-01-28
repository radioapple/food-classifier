# Food Classifier
This repository contains a convolutional neural network that can identify different foods from an image. The neural network is trained on the 
[Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The Food101 dataset contains 101 different classes of food and a total of 101,000 images. There are 750 training images and 250 testing images per class,
with a total of 1000 images per class.

The repository is split into an **experimental portion** and a portion containing the **final model**.

## Experimental Portion

The experimental portion looks at only 3 classes instead of the full 101 classes. The models in this section are built from scratch using PyTorch, and various hyperparameter values and model architectures were experimented with to see if they make a difference in the test accuracy.

## In Between
After the experimental portion, I tried the model architecture found in the experimental portion and attempted to train it on the full 101,000 images dataset, and to no one's surprise, the model was far too simple to be of any use.

## Final Model Portion

After the simple model failed, I decided to use transfer learning on a much more complex model. This resulted in the final model, which was an EfficientNetB2 model, pre-trained on ImageNet. I adjusted the classifier layer, and then trained the model on the full dataset for 25 epochs and achieved
a test accuracy of 73.2%. It also achieved at top-5 accuracy of 91.2%.

##
For more details on these sections, see the README files contained in the following sections:
* Experiment
* Experiment/Experiment 1
* Experiment/Experiment 2
* Final Model
