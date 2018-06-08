# convolutional neural networks
CMPSC 165B Machine Learning Project

Create a machine learning model to predict the Fashion MNIST data set.

This program requires numpy and tensorflow.
To run the app, run the following command in the terminal:
```
python prediction.py
```

The script will check whether a saved model exists. If it does, it skips straight to making predictions on the testing set. Otherwise, it will load the training data and fit the model. The model is a 4 layer convolution neural network.

The program outputs a .txt file of predictions for the testing data. The labels of the testing data was not provided. This model achieves an accuracy of about 90%.

Training and testing data are not pushed to github, due to their sizes.
