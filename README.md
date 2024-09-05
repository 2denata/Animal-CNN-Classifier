![](https://img.shields.io/badge/Python-red?style=for-the-badge&logo=python) 

# Animal Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of three animal categories: cat, dog, and tiger. The model is built using TensorFlow and Keras and utilizes image augmentation techniques to enhance its performance.

# Features
- Classifies images into three classes: cat, dog, and tiger.
- Uses CNN architecture with Conv2D, MaxPooling2D, Flatten, and Dense layers.
- Image augmentation techniques like zooming, shearing, and flipping to improve generalization.

# Dataset
The dataset contains images of cats, dogs, and tigers organized into separate directories for training and testing. The dataset can be downloaded from [this link](https://drive.google.com/file/d/1a_oH1olkINKoJjhngNe8aMeEi3-xSqUp/view?usp=sharing).

# Model Architecture
- **Convolutional Layers**: Extract features from images.
- **MaxPooling**: Reduces the dimensionality of feature maps.
- **Flatten**: Converts 2D matrices to a 1D vector for fully connected layers.
- **Fully Connected Layers**: Dense layers for final classification.

# Requirements
To run this project, the following packages are required:
- Python 3.5 or higher
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required libraries using:
```bash
pip install tensorflow keras numpy matplotlib
```

# How to Run
1. Clone this repository:
```bash
git clone https://github.com/your-username/Animal-Image-Classification-CNN.git
```
2. Download the dataset and place it in a datasets/ folder with the following structure:
```markdown
datasets/
    training_set/
        cats/
        dogs/
        tigers/
    test_set/
        cats/
        dogs/
        tigers/
```

3. Run the CNN script:
```bash
python CNN.py
```
The script will train the model using the provided dataset, and once trained, it will classify images from the test set.

# Model Training
The model is trained for 25 epochs with a batch size of 32. After training, the model is evaluated on the test set. The script also includes code for testing individual images for classification.

# Example Prediction
Once trained, the model can predict the class of a given image. For example, you can classify a new image of a cat, dog, or tiger by running the prediction section of the code.
![image](https://github.com/user-attachments/assets/28aa4f71-6610-4104-9671-ec8b6b70c180)
![image](https://github.com/user-attachments/assets/03acb1b0-7c38-4b15-98a4-794be094d369)
![image](https://github.com/user-attachments/assets/b0480066-2805-4eaf-bb2b-0aff11d8dbbd)



