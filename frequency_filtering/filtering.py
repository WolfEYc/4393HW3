import cv2
import numpy as np


class Filtering:

    def __init__(self, image):
        self.image = image
        self.img_mask = self.get_img_mask

    @staticmethod
    def get_img_mask(shape):

        img_mask = np.full(shape, 1, dtype=np.uint8)

        img_mask = cv2.circle(img_mask, (299, 232), 8, 0, -1)
        img_mask = cv2.circle(img_mask, (213, 283), 8, 0, -1)
        img_mask = cv2.circle(img_mask, (235, 240), 8, 0, -1)
        img_mask = cv2.circle(img_mask, (277, 270), 8, 0, -1)

        return img_mask

    @staticmethod
    def post_process_image(image):

        log = np.log(image)
        rows, cols = np.shape(log)
        fimage = np.zeros((rows, cols), dtype=np.uint8)

        minimum = log[0][0]
        maximum = log[0][0]

        for i in range(0, rows):
            for j in range(0, cols):
                if log[i][j] < minimum:
                    minimum = log[i][j]
                if log[i][j] > maximum:
                    maximum = log[i][j]

        p = 255 / (maximum - minimum)
        L = (0 - minimum) * p

        for i in range(0, rows):
            for j in range(0, cols):
                fimage[i][j] = p * log[i][j] + L

        return fimage

    def filter(self):
        shape = np.shape(self.image)
        fft = np.fft.fft2(self.image)
        fft_shifted = np.fft.fftshift(fft)

        mag_dft = self.post_process_image(np.abs(fft_shifted))
        mag_fdft = np.zeros(np.shape(mag_dft))

        self.img_mask = self.get_img_mask(shape)
        f_fft = np.zeros(shape, dtype=complex)

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                f_fft[i][j] = fft_shifted[i][j] * self.img_mask[i][j]
                mag_fdft[i][j] = mag_dft[i][j] * self.img_mask[i][j]

        inverse_shift = np.fft.ifftshift(f_fft)
        inverse_fft = np.fft.ifft2(inverse_shift)

        img = np.abs(inverse_fft)

        return [img, mag_dft, mag_fdft]