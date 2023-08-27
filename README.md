# Handwritten Number Recognition

## Overview
This project is a machine learning model that is able to recognize handwritten numbers. The model is trained on the MNIST dataset, which contains 60,000 images of handwritten digits. The model is able to recognize digits with an accuracy of over 87%.

## Requirements
- Python 3.6 or higher
- TensorFlow
- NumPy
- Matplotlib

## Usage
1. Clone the repository:
```
git clone https://github.com/Jomar77/handwritten-number-recognition.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Train the model:
```
python train.py
```
4. Test the model on new images:
```
python main.py --image path/to/image.png
```

# Note
The trained model should be able to recognize handwritten digits with an accuracy of over 98%. The results can be visualized using the plot_results.py script.

## Model
The model used in this project is a convolutional neural network (CNN) implemented using the Keras library with TensorFlow as the backend.

## Conclusion
he model architecture and training code are based on the TensorFlow 2.x official [tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
