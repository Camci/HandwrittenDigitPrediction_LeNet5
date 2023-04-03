# Handwritten Digit Prediction with LeNet-5 CNN
This application is a basic tkinter GUI application that allows users to select a handwritten digit image and get a prediction of the digit using LeNet-5 Convolutional Neural Network (CNN) model in the background.

## Getting Started

To get started with this application, you will need to have Python 3 installed on your machine. Clone the repository and install the required packages:

1. PyTorch
2. Numpy
3. OpenCV
4. PIL
5. Tkinter

***NOTE: All of them can easily install by using "pip3 install ......" command!***


Then, run the following command to launch the application:

```
python3 Predictor.py
```

## Usage

The main window of the application consists of two dropdown menus and an image viewer.

The first dropdown menu allows the user to select a digit from 0 to 9. The second dropdown menu displays the available handwritten images for the selected digit. By default, the application shows the first image of the first digit.

To make a prediction, click the "Guess" button. The application will load the selected image, resize it to 28x28 pixels, normalize it, and feed it to the LeNet-5 model for prediction. The predicted digit will be displayed in the result label.

## Architecture

The application uses LeNet-5 CNN model for digit recognition. The model was trained on the MNIST dataset and achieved an accuracy of 99.2%. The pretrained model is stored in the utils/lenet5_mnist.pt file.


## Prediction Results

The network can handle many handwritten images even it created by AI. For personal interest and testes, I generated digit numbers with using DALL-E 2 AI image generator. The result is shocked, the trained model can able to predict numbers almost perfectly.

## Credits

This application was created by Mustafa Camcı. The LeNet-5 model implementation was adapted from YANN LECUN
