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
