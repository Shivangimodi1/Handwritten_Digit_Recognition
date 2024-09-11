import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('handwritten.h5')

# Prediction of Digits
for i in range(0,10):
    img = cv2.imread(f"digits/digit_{i}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("Predicted digit is: ", {np.argmax(prediction)})
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
