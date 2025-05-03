
import os
import numpy as np
import tensorflow as tf
import streamlit as st
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate, LayerNormalization
from sklearn.metrics import classification_report
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 224
FRAME_SKIP = 5
SEQUENCE_LENGTH = 10

# Dataset paths
real_videos_path = "C:/Users/hp/Downloads/archive (17)/FF++/real"
fake_videos_path = "C:/Users/hp/Downloads/archive (17)/FF++/fake"

# Frame extraction
@st.cache_data
def extract_frames(video_path, frame_skip=FRAME_SKIP, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames, seq = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            if np.random.rand() > 0.5:
                frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            seq.append(frame)
            if len(seq) == sequence_length:
                frames.append(np.array(seq))
                seq = []
        count += 1
    cap.release()
    return np.array(frames)

# Delaunay triangulation feature extractor
def extract_delaunay_features(sequences):
    features = []
    for seq in sequences:
        del_seq = []
        for frame in seq:
            f = (frame * 255).astype(np.uint8)
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            keypoints = cv2.ORB_create().detect(gray, None)
            pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            del_count = Delaunay(pts).simplices.shape[0] if len(pts) > 3 else 0
            del_seq.append([del_count])
        features.append(del_seq)
    return np.array(features, dtype=np.float32)

# Model Definition

def create_tgnn_model():
    input_tensor = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    del_input = Input(shape=(SEQUENCE_LENGTH, 1))

    base_cnn = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_cnn.layers[-20:]:
        layer.trainable = True

    x = tf.keras.layers.TimeDistributed(base_cnn)(input_tensor)
    x = tf.keras.layers.TimeDistributed(GlobalAveragePooling2D())(x)
    x = Concatenate()([x, del_input])

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
    attention = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Concatenate()([x, attention])

    x = tf.keras.layers.Flatten()(x)
    x = LayerNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_tensor, del_input], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_tgnn_model()

# Streamlit App
st.title("Deepfake Detection using TGNN")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])


if uploaded_file:
    st.video(uploaded_file)
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Extracting frames and preparing sequences...")
    sequences = extract_frames("temp_video.mp4")
    del_features = extract_delaunay_features(sequences)

    st.write("Running deepfake detection on each sequence...")
    preds = model.predict([sequences, del_features])
    avg_pred = np.mean(preds)
    label = "Real" if avg_pred < 0.5 else "Fake"

    st.write(f"### Prediction: {label} ({avg_pred:.2f})")

    fig, ax = plt.subplots()
    ax.plot(preds, label='Sequence Predictions')
    ax.axhline(0.5, color='red', linestyle='--', label='Threshold')
    ax.set_title("Per-sequence Deepfake Prediction")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Confidence")
    ax.legend()
    st.pyplot(fig)  
