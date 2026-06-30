import cv2
import numpy as np
import tensorflow as tf

labels = [
    "A","B","C","D","E",
    "F","G","H","I","J"
]

model = tf.keras.models.load_model("sign_model.h5", compile=False)

print("Model Loaded Successfully!")

def predict_image(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)

    return labels[index]