import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


# Function to extract patches from an image given keypoints
# and a patch size. The function returns the patches and their coordinates.
def extract_patches(im, kp, patch_size):
    half_size = patch_size // 2
    patches = []
    valid_coords = []

    for y, x in kp:
        if (
            y - half_size >= 0
            and y + half_size < im.shape[0]
            and x - half_size >= 0
            and x + half_size < im.shape[1]
        ):
            patch = im[
                y - half_size : y + half_size + 1, x - half_size : x + half_size + 1
            ]
            patches.append(patch)
            valid_coords.append((y, x))

    return np.array(patches), valid_coords


# This function displays a specified number of random patches from the list of patches.
def show_random_patches_with_keypoints(patches, patch_size, num_show=5):
    half_size = patch_size // 2
    plt.figure(figsize=(12, 3))

    indices = random.sample(range(len(patches)), min(num_show, len(patches)))

    for i, idx in enumerate(indices):
        patch = patches[idx]
        plt.subplot(1, num_show, i + 1)
        plt.imshow(patch, cmap="gray")
        plt.scatter(half_size, half_size, c="r", s=20)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Function to extract fixed-size patches around key-points
def extract_patch(image, keypoint, patch_size=16):

    x, y = keypoint.pt
    half_patch = patch_size // 2

    x_start = max(int(y - half_patch), 0)
    y_start = max(int(x - half_patch), 0)
    x_end = min(int(y + half_patch), image.shape[0])
    y_end = min(int(x + half_patch), image.shape[1])

    patch = image[x_start:x_end, y_start:y_end]

    # Pad the patch with zeros if it is smaller than the patch_size (at the image edges)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        patch = cv2.copyMakeBorder(
            patch,
            0,
            patch_size - patch.shape[0],
            0,
            patch_size - patch.shape[1],
            cv2.BORDER_CONSTANT,
            value=0,
        )

    return patch
