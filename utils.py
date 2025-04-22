import numpy as np
import cv2

# Sobel kernels
Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = Sx.T


def compute_gradients(patch):
    dx = cv2.filter2D(patch, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(patch, ddepth=-1, kernel=Sy)
    return dx, dy
