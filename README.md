# Bee Sound Analysis for Swarming Behavior Detection
# Buzz 2 Dataset

## Overview
The Buzz 2 dataset is designed for studying the behavior of bees in both swarming and non-swarming conditions. It consists of a training set, testing set, and validation set, each with different scenarios to ensure diversity in the data.

## Data Collection

Mrs. Hong collaborated with Bee Professionals for six months to collect a comprehensive dataset. This dataset includes audio recordings of various bee activities, with a particular emphasis on swarming behavior. The collected data serves as the foundation for training and testing the models.
## Dataset Details


### Training Set
The training set contains records of bees when they are non-swarming on different days. It includes data from various recorders to enhance diversity.

| Status       | Number of Samples |
|--------------|-------------------|
| None Swarming| 12,659            |
| Swarming     | 12,073            |

### Testing Set
The testing set includes two scenarios of bee status and recorders, isolated from the training set in terms of days.

| Status       | Number of Samples |
|--------------|-------------------|
| None Swarming| 5,000             |
| Swarming     | 6,575             |

### Validation Set
Similar to the testing set, the validation set contains two scenarios of bee status and recorders, isolated from the training set in terms of days.

| Status       | Number of Samples |
|--------------|-------------------|
| None Swarming| 2,000             |
| Swarming     | 2,249             |
## Feature Extraction Methods

### 1. Mel-Frequency Cepstral Coefficients (MFCCs)
**Description:**
MFCCs are a set of coefficients that represent the short-term power spectrum of a sound signal.
- Total time elapsed for Training: 366.22 seconds
- Total time elapsed for Validation: 61.65 seconds
- Total time elapsed for Testing: 228.59 seconds

### 2. Short-Time Fourier Transform (STFT)

**Description:**
STFT is a method for analyzing the frequency content of a signal over time. 
- Total time elapsed for Training: 268.29 seconds
- Total time elapsed for Validation: 31.55 seconds
- Total time elapsed for Testing: 128.45 seconds
### 3. Chroma

**Description:**
Chroma features represent the energy distribution of pitch classes in an audio signal.
- Total time elapsed for Training: 450.26 seconds
- Total time elapsed for Validation: 85.87 seconds
- Total time elapsed for Testing: 292.96 seconds

# Distribution of Training Data

### MFCCs ( 80 Features )
![PCA Plot](https://github.com/ThinhHoang1/Swarming-Bee-Detection-On-Buzz-1/blob/Buzz-1/Data%20Distribustion/MFCCs%20Distribusion/MFCCs%20(80%20Features%20)%20Train%20Set.png)

### STFT (1024 Features ) 
![PCA Plot](https://github.com/ThinhHoang1/Swarming-Bee-Detection-On-Buzz-1/blob/Buzz-1/Data%20Distribustion/STFT%20Distribution/STFT%20(%201024%20Features%20)%20Train%20Set.png)

### Chroma ( 24 Features ) 
![PCA Plot](https://github.com/ThinhHoang1/Bee-Sound-Analysis-for-Swarming-Behavior-Detection/blob/Buzz_2/Data%20Distribustion/Chroma%20Distribution/PCA%20Visualization%20of%20Training%20Data%20(2D).png)

## Model Training for Deep learning model

This section provides details about the training process for the model in the context of GitHub. The primary focus is on preventing overfitting through the use of early stopping. The EarlyStopping callback from the Keras library is employed to monitor the validation loss during training. If no improvement is observed after 20 epochs, the training process will be halted. Additionally, the model will restore the best weights based on the validation loss.

```python
earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
epochs = 100
batch_size = 64
current_time = int(time.time())
stop_training_callback = earlystopper
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_data=(X_val, y_val), 
                    callbacks=[stop_training_callback])
print("Training completed in {} seconds.".format(int(time.time()-current_time)))
```

## Model Performance on Buzz 1: 



| Model                | Accuracy using MFCCs | Accuracy using STFT |Accuracy using Chroma |          
|----------------------|----------------------|---------------------|----------------------|                   
| Random Forest        | 99%                  |  57%                |   95%                |
| SVM                  | 83%                  |  84%                |   94%                |
| KNN                  | 99%                  |  81%                |   92%                |
| Logistic Regression  | 93%                  |  82%                |   91%                |


|   Model (1D_CNN )     | Training Accuracy  | Validation Accuracy  | Testing Accuracy |
|-----------------------|--------------------|----------------------|------------------|
| Using MFCCs           | 100%               | 100%                 |         97%      |
|                       |                    |                      |                  |
|Using STFT             | 100%               | 100%                 |        100%      |
|                       |                    |                      |                  |
|Using Chroma           | 62%                |  38%                 |    66%           | 
                 
