import tensorflow as tf
from tensorflow.keras.metrics import  Precision, Recall, BinaryAccuracy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

def testing(test, model):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())
    img = cv2.imread('../data/ak47/61cGUPGlRZL.jpg')
    plt.imshow(img)
    plt.show()
    resize = tf.image.resize(img, (256, 256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()
    # predicting

    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        print(f'Predicted class is AK47')
    else:
        print(f'Predicted class is Mp5')

#     saving model

    model.save(os.path.join('../models', 'imageclassifier.h5'))
    new_model = load_model('imageclassifier.h5')
    new_model.predict(np.expand_dims(resize / 255, 0))




