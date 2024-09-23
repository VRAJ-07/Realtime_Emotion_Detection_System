# Realtime Emotion Detection System

Data Set Link - https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

## Overview

This repository hosts a **Realtime Emotion Detection System** that identifies facial emotions using **Convolutional Neural Networks (CNNs)**. The model is trained to recognize 7 distinct emotions: **Anger, Disgust, Fear, Happy, Neutral, Sad, and Surprise** using the FER (Facial Expression Recognition) dataset. The system can be used for real-time emotion detection from images or video feeds.

---

## Features

- **Data Loading & Visualization**: The dataset consists of grayscale images, each resized to 48x48 pixels for processing. The images are visualized in a grid layout to get an initial overview.
  
- **Data Preprocessing**: Preprocessing is handled by the `ImageDataGenerator` to facilitate real-time data augmentation during training and validation.

- **Model Architecture**: 
  - The CNN architecture consists of 4 convolutional layers followed by fully connected layers.
  - **Batch Normalization** and **Dropout** are employed to improve generalization.
  - **Max Pooling** is used to reduce dimensionality after each convolution layer.

- **Training & Validation**: 
  - The model is trained on the FER dataset with an **Adam optimizer** and **categorical crossentropy** as the loss function.
  - Training is monitored using **early stopping** to avoid overfitting, and the learning rate is adjusted dynamically using **ReduceLROnPlateau**.

- **Accuracy & Loss Visualization**: After training, plots are generated to show the evolution of training and validation accuracy and loss across epochs.

---

## Project Structure

The key components of this project are:

- **Dataset**: FER dataset structured with training and test sets, each divided into 7 emotion categories.
  
- **Model**: A CNN model built using the Keras API with 4 convolutional layers and fully connected dense layers at the end.

- **Training and Validation**: The model is trained using the training set and validated using the test set, with accuracy and loss metrics tracked over time.

- **Callbacks**: 
  - **Early Stopping**: Stops training if the model performance doesn’t improve for a set number of epochs.
  - **Model Checkpoint**: Saves the best model during training based on validation accuracy.
  - **ReduceLROnPlateau**: Dynamically reduces the learning rate when the validation loss plateaus.

---

## Model Performance

- The model was trained on approximately 28,000 images belonging to 7 classes and validated on 7,000 images.
- After training for 14 epochs, the model reached a validation accuracy of around 60%.
- Further improvements to model performance could be achieved with additional training or more sophisticated architecture adjustments.

---

## Requirements

To run this project, the following dependencies are required:

- **Python 3.x**
- **TensorFlow/Keras**: For deep learning model construction and training.
- **Numpy**: For numerical operations.
- **Pandas**: For data handling.
- **Matplotlib & Seaborn**: For plotting graphs of accuracy and loss metrics.
- **OpenCV**: (Optional) For real-time emotion detection from live video feeds.

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Steps to Run the Project

1. **Clone the Repository**:
   - Clone the repository to your local system using:
   ```bash
   git clone https://github.com/VRAJ-07/Realtime_Emotion_Detection_System.git
   cd Realtime_Emotion_Detection_System
   ```

2. **Download the Dataset**:
   - Download the FER dataset and place it in the appropriate directory (structured with 'train' and 'test' subfolders inside an `images/` folder).

3. **Train the Model**:
   - Execute the training script to train the model on the FER dataset.
   - The best model weights will be saved automatically upon completion.

4. **Evaluate the Model**:
   - After training, you can load the saved model and use it for predictions on new images or real-time video streams.

5. **Visualize Results**:
   - Plot the accuracy and loss metrics for training and validation to assess model performance.

---

## Future Work

- **Real-time Integration**: The model can be integrated with OpenCV to detect emotions in real-time from a camera feed.
  
- **Model Improvements**: Experimenting with transfer learning (e.g., using pre-trained models like VGG16, ResNet) or trying different architectures could improve model accuracy.
  
- **Hyperparameter Tuning**: Tuning parameters such as batch size, learning rate, and optimizer selection could further enhance the model’s performance.
