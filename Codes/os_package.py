import cv2
import imghdr
import os
# import tensorflow as tf
from dodgy_images import preprocess_dodgy_image

data_directory = 'data'
file_extensions = ['jpg', 'jpeg', 'png']
# print(file_extensions[2])

# print(os.listdir(data_directory))
#
# # displaying the content inside the happy directory
# folder_path = os.listdir(os.path.join(data_directory, 'happy'))
# # print(folder_path)
#
# # iterating through the directory using os package
# happy_path = os.path.join(data_directory,'happy')
# happy1_path = os.path.join(happy_path, 'happy1')
# # print(os.listdir(happy1_path))
#
# # finding the absolute path using os package
# happy1_apath = os.path.abspath(happy1_path)
# # print(happy1_apath)
#
# # getting image size
# img = os.path.join('data', 'happy', '35438_hd.jpg')
# img_path = os.path.abspath(img)
# print(img_path)
# # print(img.shape)
# image1 = cv2.imread(img_path)
# cv2.imshow('hello', image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()