import matplotlib.pyplot as plt
import tensorflow as tf


def load_data(data_dir):
    # creating the data pipeline
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    # changing the permission to access the data
    data_iterator = data.as_numpy_iterator()
    # accessing the data
    batch = data_iterator.next()
    # Checking the shape of the image after converting into dataset
    print(batch[0].shape)
    # printing the labels
    print(batch[1])
    # this return the value 2 which means images and its labels in tuple form
    print(len(batch))
    # preprocess part
    data = data.map(lambda x, y: (x / 255, y))
    # reading the data for scaling
    scaled_iterator = data.as_numpy_iterator()
    batch = scaled_iterator.next()
    print(batch[0].min())
    print(batch[0].max())
    print(len(data))
    return  data

# def preprocess_data(data):  # we can add the batch parameter here and uncomment the below lines.
    # print(batch[0].min())
    # print(batch[1].max())
    # # Scaling image huge array to 0-1 range.
    # scaled = batch[0] / 255
    # print(scaled.min())
    # print(scaled.max())
    # mapping the data by normalizing


# splitting dataset
def split_dataset(data):
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    print(len(train), len(val), len(test))
    return train, val, test
