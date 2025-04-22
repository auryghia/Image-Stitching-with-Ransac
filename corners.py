from utils import compute_gradients
import numpy as np
import cv2


Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

Sy = Sx.T

G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


def corner_response(image, k, Sx=Sx, Sy=Sy, G=G):
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    A = cv2.filter2D(dx * dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy * dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx * dy, ddepth=-1, kernel=G)

    return (A * B - (C * C)) - k * (A + B) * (A + B)
