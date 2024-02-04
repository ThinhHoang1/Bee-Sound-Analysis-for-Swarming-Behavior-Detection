# Bee Sound Analysis for Swarming Behavior Detection

## Overview

This project focuses on the identification of swarming bee behavior using audio features extracted from recordings. The audio features include Mel-Frequency Cepstral Coefficients (MFCCs), Short-Time Fourier Transform (STFT), and Chroma. The goal is to leverage machine learning and deep learning techniques to classify different bee sounds and understand their swarming patterns.

## Data Collection

Mrs. Hong collaborated with Bee Professionals for six months to collect a comprehensive dataset. This dataset includes audio recordings of various bee activities, with a particular emphasis on swarming behavior. The collected data serves as the foundation for training and testing the models.

# Buzz 1: Bee Sound Classification Dataset

## Data Set

The dataset is divided into training, testing, and validation sets, with careful consideration for different factors such as days, bee status, and recorder variations.

|                   | Training | Testing | Validation |
|-------------------|----------|---------|------------|
| Non-Swarming      | 12,728   | 6,000   | 3,000      |
| Swarming          | 10,656   | 6,773   | 2,641      |


## Notes
- **Days:** Isolation of sets is based on different days.
- **Bee Status:** Non-swarming and swarming instances are separated.
- **Recorder Variation:** Different recorders are used for swarming instances.

## Feature Extraction Methods

### 1. Mel-Frequency Cepstral Coefficients (MFCCs)
**Description:**
MFCCs are a set of coefficients that represent the short-term power spectrum of a sound signal.
- Total time elapsed for Training: 325.88 seconds
- Total time elapsed for Validation: 89.27 seconds
- Total time elapsed for Testing: 247.53 seconds
### 2. Short-Time Fourier Transform (STFT)

**Description:**
STFT is a method for analyzing the frequency content of a signal over time. 
- Total time elapsed for Training: 288.77 seconds
- Total time elapsed for Validation: 89.24 seconds
- Total time elapsed for Testing: 247.53 seconds
### 3. Chroma

**Description:**
Chroma features represent the energy distribution of pitch classes in an audio signal.
- Total time elapsed for Training: 256.58 seconds
- Total time elapsed for Validation: 71.75 seconds
- Total time elapsed for Testing: 192.11 seconds

# Distribution of Training Data

### MFCCs ( 80 Features )
![PCA Plot](https://github.com/ThinhHoang1/Swarming-Bee-Detection-On-Buzz-1/blob/Buzz-1/Data%20Distribustion/MFCCs%20Distribusion/MFCCs%20(80%20Features%20)%20Train%20Set.png)

### STFT (1024 Features ) 
![PCA Plot](https://github.com/ThinhHoang1/Swarming-Bee-Detection-On-Buzz-1/blob/Buzz-1/Data%20Distribustion/STFT%20Distribution/STFT%20(%201024%20Features%20)%20Train%20Set.png)

### Chroma ( 24 Features ) 
![PCA Plot](https://github.com/ThinhHoang1/Swarming-Bee-Detection-On-Buzz-1/blob/Buzz-1/Data%20Distribustion/Chroma%20Distribution/Chroma%20(%2024%20Features%20)%20Train%20set.png)

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
| Random Forest        | 98%                  |  65%                |   54%                |
| SVM                  | 92%                  |  84%                |   59%                |
| KNN                  | 96%                  |  86%                |   63%                |
| Logistic Regression  | 93%                  |  93%                |   63%                |


|   Model (1D_CNN )     | Training Accuracy  | Validation Accuracy  | Testing Accuracy |
|-----------------------|--------------------|----------------------|------------------|
| Using MFCCs           | 100%               | 99%                  |         97%      |
|                       |                    |                      |                  |
|Using STFT             | 99%                | 96%                  |       90%        |
|                       |                    |                      |                  |
|Using Chroma           | 85%                |  76%                 |    59%           | 
                 
