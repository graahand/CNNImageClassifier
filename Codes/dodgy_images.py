import os
import cv2
import imghdr

data_directory = 'data'
file_extensions = ['jpg', 'jpeg', 'png']


def preprocess_dodgy_image(data_dir):
    for sub_directory in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, sub_directory)):
            image_path = os.path.join(data_dir, sub_directory, image)
            try:
                img = cv2.imread(image_path)
                type_of_image = imghdr.what(image_path)
                if type_of_image not in file_extensions:
                    print("Image not in extension list {}.".format(image_path))
                    os.remove(image_path)
            except FileNotFoundError as e:
                print('Issue while loading the image {}.'.format(image_path))






