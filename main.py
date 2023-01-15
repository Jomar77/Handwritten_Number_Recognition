import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model('handWritten.model')

_img = 1
while os.path.isfile(f'digits/digit{_img}.png'):
    try:
        
        img = cv2.imread(f'digits/digit{_img}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Prediction: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(str(e))
    finally:
        _img += 1
    





