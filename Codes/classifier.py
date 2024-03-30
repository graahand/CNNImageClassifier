import os
from dodgy_images import preprocess_dodgy_image
from gpu_or_cpu import gpu_or_cpu
from os_package import data_directory
from load_data import load_data
from load_data import split_dataset
from deeplearning_model import neural_network, plot_performance
from testing_model import testing
# gpu_or_cpu
gpu_or_cpu()

# using the dodgy_image_script
preprocess_dodgy_image(data_directory)

# loading the dataset
data = load_data(data_directory)
# print(data, batch)

# splitting the dataset
train, val, test = split_dataset(data)

model = neural_network(train, val, test)

testing(test, model)
