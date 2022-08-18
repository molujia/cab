import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np


def showRate(name):
    image = cv.imread('chars' + os.sep + name, cv.IMREAD_GRAYSCALE)
    line = np.sum(image, axis=0)
    max_ = np.max(line)
    line = np.around(line / max_, decimals=2)
    plt.xlabel('x')
    plt.ylabel('rate')
    plt.scatter(range(len(line)), line)
    for x, y in zip(range(len(line)), line):
        plt.text(x, y, (x, y), ha='center', va='bottom', fontsize=10)
    plt.show()
    cv.imshow('image', image)
    cv.waitKey()
    cv.destroyAllWindows()
    pass


if __name__ == '__main__':
    showRate('5_3_2.png')
