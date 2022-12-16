import numpy as np
import math


class Dft:
    def __init__(self):
        pass

    @staticmethod
    def forward_transform(matrix):

        shape = np.shape(matrix)
        N = shape[0]
        fft = np.zeros((N, N), dtype=complex)

        for u in range(0, N):
            for v in range(0, N):
                for i in range(0, N):
                    for j in range(0, N):
                        fft[u][v] += matrix[i][j] * (math.cos((2 * math.pi / N) * (u * i + v * j)) - 1j * math.sin(
                            (2 * math.pi / N) * (u * i + v * j)))

        return fft

    @staticmethod
    def inverse_transform(matrix):

        shape = np.shape(matrix)
        N = shape[0]
        inverse = np.zeros((N, N), dtype=complex)

        for i in range(0, N):
            for j in range(0, N):
                for u in range(0, N):
                    for v in range(0, N):
                        inverse[i][j] += matrix[u][v] * (math.cos((2 * math.pi / N) * (u * i + v * j)) + 1j * math.sin(
                            (2 * math.pi / N) * (u * i + v * j)))

        return inverse

    @staticmethod
    def magnitude(matrix):

        shape = np.shape(matrix)
        N = shape[0]

        mag = np.zeros((N, N))

        for u in range(0, N):
            for v in range(0, N):
                mag[u][v] = math.sqrt(math.pow(np.real(matrix[u][v]), 2) + math.pow(np.imag(matrix[u][v]), 2))

        return mag
