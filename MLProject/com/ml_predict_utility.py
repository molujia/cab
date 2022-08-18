import cv2 as cv
import numpy as np

def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image_path, width, height):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)