# Food Classifier
This repository contains code for training, testing, and using a convolutional neural network (CNN) for identifying the food present in an image. The neural network is trained on the 
[Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The Food101 dataset contains 101 different classes of food and a total of 101,000 images. There are 750 training images and 250 testing images per class,
with a total of 1000 images per class.

The model used extends EfficientNetB2 (pre-trained on ImageNet) by adding a classification layer. For training, the EfficientNetB2 weights are frozen and only the classification layer is trained. The final model produces a **top-1 accuracy of 73.2%**, with a **top-5 accuracy of 91.2%**.

The code used for training can be found at [Final Model / training_model.ipynb](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/training_model.ipynb). The trained model's weights can be found at [Final Model / effnetb2_model_5_results.zip](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/effnetb2_model_5_results.zip). The code for loading in the saved model, testing, and classifying new images can be found at [Final Model / loading_and_testing_model.ipynb](https://github.com/radioapple/food-classifier/blob/main/Final%20Model/loading_and_testing_model.ipynb)

## Results
The model achieves a top-1 accuracy of 73.2%, with a top-5 accuracy of 91.2%.

### Confusion Matrix
<img width="1833" height="1790" alt="image" src="https://github.com/user-attachments/assets/6193fab0-f947-4d89-8534-ead7e4d7f91a" />

### Random Sample of 25 Images from Test Set
<img width="1949" height="1989" alt="image" src="https://github.com/user-attachments/assets/56c5676f-8a97-4ac2-946d-ffb60eb3b702" />

Some of the incorrect classifications from the model may be understandable (e.g., the ceaser salad, risotto, chicken wings, and chocolate mousse). In the case of the huevos rancheros, the dish contains eggs, so the model may have learned "something that looks flat + there are eggs = omlette". However, there are also cases where the model seems entirely off, such as predicting hummus when the image shows samosas. Even in the wrong but understandable cases, many appear visually distinguishable to a human observer. This suggests that the model could likely be improved further, as the errors do not seem to stem purely from ambiguity in the data.

## Notes  
- A manual train/test split was used rather than the official Food-101 split.  
- The file paths in the notebooks are currently set up for the Kaggle environment I used during development and would need to be adjusted to run locally from this repository.
- The [`Modules`](https://github.com/radioapple/food-classifier/tree/main/Modules) folder contains earlier experimental PyTorch implementations of a smaller model. The previous experimental results have been removed, but the code is retained for reference.
- The current implementation uses only a training and test set.

## Next Steps  
- The learning rate was manually adjusted every 5 epochs, and training was stopped after 22 epochs. Future work could experiment with learning rate scheduling and stopping rules.
