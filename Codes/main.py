import tensorflow as tf
import cv2
import imghdr
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Check if TensorFlow can access GPU devices
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Remove images with invalid extensions
data_dir = '../data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Removing image with invalid extension:", image_path)
                os.remove(image_path)
        except Exception as e:
            print("Error processing image:", e)

# Load and visualize the images
data = tf.keras.utils.image_dataset_from_directory(
    'data',
    labels='inferred',
    label_mode='int',  # or 'categorical' for one-hot encoding
    batch_size=32,
    image_size=(224, 224),
    shuffle=True
)

# Fetch a batch of images
data_iterator = iter(data)
images, labels = next(data_iterator)

# Visualize the images
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(images[:4]):
    ax[idx].imshow(img.numpy().astype("uint8"))
    ax[idx].set_title(labels[idx].numpy())
plt.show()

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
#
# print(train_size)
# print(val_size)
# print(test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)



model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(224, 224, 3)))  # Adjusted input shape
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
print(hist)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
# plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()