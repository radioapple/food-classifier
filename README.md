# Food Classifier
This repository contains code for training, testing, and using a convolutional neural network (CNN) for identifying the food present in an image. The neural network is trained on the 
[Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The Food101 dataset contains 101 different classes of food and a total of 101,000 images. There are 750 training images and 250 testing images per class,
with a total of 1000 images per class.

The model used extends EfficientNetB2 (pre-trained on ImageNet) by adding a classification layer. For training, the EfficientNetB2 weights are frozen and only the classification layer is trained. The final model produces a top-1 accuracy of 73.2%, with a top-5 accuracy of 91.2%.

The code used for training can be found at [Final Model / training_model.ipynb](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/training_model.ipynb). The trained model's weights can be found at [Final Model / effnetb2_model_5_results.zip](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/effnetb2_model_5_results.zip). The code for loading in the saved model, testing, and classifying new images can be found at [Final Model / loading_and_testing_model.ipynb](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/loading_and_testing_model.ipynb)


## Notes
- I used a manual train test split rather than the official train/test split.
- The current paths are based off the kaggle environment I worked in and would need to be adjusted to reference locations inside the repository later.
- The [Modules](https://github.com/radioapple/food-classifier/tree/main/Modules) folder contains old code that I used when experimenting with manually setting up a smaller model using PyTorch. The old results and experiments have been removed but the code is still up for future referencing.

## Next Steps
- The learning rate was manually changed every 5 epochs in the current code and stopped manually after 22 epochs. Future work could experiment with learning rate scheduling and stopping rules.
