import numpy as np
import math


class Filtering:

    def __init__(self, image):
        self.image = image

    @staticmethod
    def get_gaussian_filter():

        img_filter = np.zeros((5, 5))
        sigma = 1

        for i in range(0, 5):
            for j in range(0, 5):
                y = i - 2
                x = j - 2
                img_filter[i][j] = (1 / (2 * math.pi * sigma * sigma)) * math.exp(
                    -((x * x + y * y) / 2 * sigma * sigma))

        return img_filter

    @staticmethod
    def get_laplacian_filter():
        return [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

    def perform_gaussian_filter(self):
        shape = np.shape(self.image)
        filter_img = np.zeros((shape[0], shape[1]))
        padding = 4
        img_filter = self.get_gaussian_filter()
        image_padding = np.zeros((shape[0] + padding * 2, shape[1] + padding * 2))

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                image_padding[i + padding][j + padding] = self.image[i][j]
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                y = i + padding
                x = j + padding
                intensity = 0
                for m in range(0, padding + 1):
                    for n in range(0, padding + 1):
                        intensity += img_filter[m][n] * image_padding[y - (padding - m)][x - (padding - n)]
                filter_img[i][j] = intensity

        return filter_img

    def perform_laplacian_filter(self):
        shape = np.shape(self.image)
        filter_img = np.zeros((shape[0], shape[1]))
        padding = 2
        img_filter = self.get_laplacian_filter()
        image_padding = np.zeros((shape[0] + padding * 2, shape[1] + padding * 2))

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                image_padding[i + padding][j + padding] = self.image[i][j]

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                y = i + padding
                x = j + padding
                intensity = 0
                for m in range(0, padding + 1):
                    for n in range(0, padding + 1):
                        intensity += img_filter[m][n] * image_padding[y - (padding - m)][x - (padding - n)]
                filter_img[i][j] = intensity

        return filter_img

    def filter(self, filter_name):
        match filter_name:
            case "gaussian":
                return self.perform_gaussian_filter()
            case "laplacian":
                return self.perform_laplacian_filter()
