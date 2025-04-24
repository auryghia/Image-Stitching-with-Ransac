from skimage.feature import corner_harris, corner_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt

Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

Sy = Sx.T

G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


def harris_corners(
    image, sigma=1, k=0.005, min_distance=1, size=10, plot=False, image_name="Image"
):
    """
    Harris corner detection using skimage's corner_harris and corner_peaks.
    Parameters:
    - image: Input image (grayscale).
    - sigma: Gaussian filter standard deviation.
    - k: Harris detector free parameter.
    - min_distance: Minimum distance between detected corners.
    - size: Size of the Gaussian filter.
    - plot: If True, plot the detected corners.
    - image_name: Name of the image for plotting title.
    """

    harris_response = corner_harris(image, sigma=sigma)
    keypoints = corner_peaks(
        harris_response,
        exclude_border=size // 2,
        min_distance=min_distance,
        threshold_rel=k,
    )

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.plot(keypoints[:, 1], keypoints[:, 0], "r.", markersize=2)
        ax.set_title(f"Harris Corners - {image_name}")
        plt.show()

    return keypoints


def corner_response(image, k, Sx=Sx, Sy=Sy, G=G):
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    A = cv2.filter2D(dx * dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy * dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx * dy, ddepth=-1, kernel=G)

    return (A * B - (C * C)) - k * (A + B) * (A + B)
