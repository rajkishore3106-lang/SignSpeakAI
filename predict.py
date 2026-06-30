import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Labels
labels = [
    "A","B","C","D","E",
    "F","G","H","I","J"
]

# Create same model architecture
model = Sequential([
    tf.keras.layers.Input(shape=(64,64,3)),

    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(),

    Flatten(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(10,activation="softmax")
])

# Load trained weights
model.load_weights("sign_weights.weights.h5")

print("Model Loaded Successfully!")

def predict_image(image):

    image = cv2.resize(image,(64,64))

    image = image.astype(np.float32)/255.0

    image = np.expand_dims(image,axis=0)

    prediction = model.predict(image,verbose=0)

    index=np.argmax(prediction)

    return labels[index]