# 🎧 Speech Emotion Recognition Web App

A fully functional web application built using **Streamlit** that classifies emotions from `.wav` audio files using a deep learning model trained on the **RAVDESS** dataset with MFCC features.

---

## 📌 Project Description

This project focuses on detecting human emotions from speech using audio processing and deep learning. The web app receives an input audio file in `.wav` format, extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features, and uses a pre-trained neural network to predict the corresponding emotion.

---

## 🧪 Dataset: RAVDESS

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- 8 emotion classes:  
  - `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Contains recordings by 24 actors (12 male, 12 female) with variations in emotion and intensity.

---

## 🔍 Preprocessing Methodology

1. **File Loading**: `.wav` files loaded using `librosa`.
2. **MFCC Extraction**:
   - Duration: 3 seconds (starting from 0.5s offset)
   - 40 MFCCs extracted using `librosa.feature.mfcc()`
   - Feature shape: `(40,)`, averaged across time frames
3. **Label Encoding**:
   - Emotion labels parsed from filenames
   - One-Hot Encoding for multi-class classification

---

## 🧠 Model Architecture

A hybrid CNN + BiGRU model trained on MFCC features with the following structure:

```python
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(180, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(512, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Bidirectional(GRU(256, return_sequences=True)),
    Bidirectional(GRU(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(8, activation='softmax')
])
```

## 📈 Classification Report

```python
                precision    recall  f1-score   support

       angry       0.88      0.84      0.86        75
        calm       0.83      0.92      0.87        75
     disgust       0.84      0.67      0.74        39
     fearful       0.76      0.67      0.71        75
       happy       0.86      0.73      0.79        75
     neutral       0.73      0.92      0.81        38
         sad       0.75      0.75      0.75        75
   surprised       0.73      0.97      0.84        39

    accuracy                           0.80       491
   macro avg       0.80      0.81      0.80       491
weighted avg       0.80      0.80      0.80       491

```

