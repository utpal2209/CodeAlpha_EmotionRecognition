import os
import librosa
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow Version:", tf.__version__)

# =========================================================
# Dataset Path
# =========================================================

dataset_path = "dataset"

# =========================================================
# Emotion Labels
# =========================================================

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# =========================================================
# Feature Extraction Function
# =========================================================

def extract_feature(file_name):

    try:

        audio, sample_rate = librosa.load(
            file_name,
            sr=None,
            res_type='kaiser_fast'
        )

        # Extract 40 MFCC Features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Take Mean
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled

    except Exception as e:

        print("Error encountered while parsing file:", file_name)
        print(e)

        return None

# =========================================================
# Load Dataset
# =========================================================

features = []
labels = []

for subdir in os.listdir(dataset_path):

    subdir_path = os.path.join(dataset_path, subdir)

    if os.path.isdir(subdir_path):

        for file in os.listdir(subdir_path):

            if file.endswith(".wav"):

                file_path = os.path.join(subdir_path, file)

                feature = extract_feature(file_path)

                if feature is not None:

                    features.append(feature)

                    # Extract Emotion Code
                    parts = file.split("-")

                    emotion_code = parts[2]

                    emotion_label = emotion_dict.get(
                        emotion_code,
                        "unknown"
                    )

                    labels.append(emotion_label)

print("\nDataset Loaded Successfully!")
print("Total Audio Files:", len(features))

# =========================================================
# Convert to NumPy Arrays
# =========================================================

X = np.array(features)
y = np.array(labels)

# =========================================================
# Feature Scaling (IMPORTANT)
# =========================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================================================
# Encode Labels
# =========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# =========================================================
# Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# Build Neural Network Model
# =========================================================

model = Sequential([

    Input(shape=(X.shape[1],)),

    Dense(256, activation='relu'),

    Dropout(0.3),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(len(np.unique(y)), activation='softmax')

])

# =========================================================
# Compile Model
# =========================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# Show Model Summary
# =========================================================

print("\nModel Summary:\n")

model.summary()

# =========================================================
# Early Stopping
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================================================
# Train Model
# =========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================================================
# Evaluate Model
# =========================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# =========================================================
# Save Model
# =========================================================

model.save("emotion_model.keras")

print("\nModel Training Completed!")
print("Model Saved Successfully!")