import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin_min
import cv2
import matplotlib.pyplot as plt


class Ransac:
    """
    RANSAC algorithm for estimating a translation matrix between two sets of keypoints.
    The algorithm iteratively samples a subset of matches, computes the translation matrix,
    and counts the number of inliers that fit the model within a specified threshold.
    The best translation matrix is the one with the most inliers.

    """

    def __init__(
        self, n_iterations=1000, threshold=1, threshold_matches=0.5, sample_size=3
    ):
        """
        Initialize the RANSAC algorithm.
        :param n_iterations: Number of iterations for RANSAC.
        :param threshold: Distance threshold for considering a point as an inlier.
        :param threshold_matches: Distance threshold for considering a match as valid.
        :param sample_size: Number of points to sample for each iteration.
        """
        # Initialize parameters

        self.n_iterations = n_iterations
        self.threshold = threshold
        self.sample_size = sample_size
        self.threshold_matches = threshold_matches
        self.inliers = []
        self.outliers = []
        self.residuals = []
        self.translation_matrix = None
        self.average_residual_inliers = None
        self.matches = None

    def compute_translation_horizontal(self, src_pts, dst_pts):
        """
        Compute the translation matrix based on the horizontal translation between two sets of points.
        :param src_pts: Source points.
        :param dst_pts: Destination points.
        :return: Translation matrix.
        """

        src_x = np.array([pt[0] for pt in src_pts])
        dst_x = np.array([pt[0] for pt in dst_pts])
        t_x = np.mean(dst_x - src_x)
        translation_matrix = np.array([[1, 0, t_x], [0, 1, 0]])

        return translation_matrix

    def apply_translation(self, pts, translation_matrix):
        """
        Apply the translation matrix to a set of points.
        :param pts: Points to be translated.
        :param translation_matrix: Translation matrix.
        :return: Translated points.
        """
        pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
        translated_pts = np.dot(pts_homogeneous, translation_matrix.T)
        return translated_pts[:, :2]

    def mutual_match(self, desc1, desc2):
        """
        Compute mutual matches between two sets of descriptors.
        A match is considered mutual if the closest descriptor in one set is also the closest in the other set.
        :param desc1: First set of descriptors.
        :param desc2: Second set of descriptors.
        :return: Dictionary of mutual matches.
        """
        # Compute pairwise distances between descriptors
        matches_1_to_2, _ = pairwise_distances_argmin_min(
            desc1, desc2, metric="euclidean"
        )
        matches_2_to_1, _ = pairwise_distances_argmin_min(
            desc2, desc1, metric="euclidean"
        )

        mutual_matches = {}
        for i, match_1 in enumerate(matches_1_to_2):

            if matches_2_to_1[match_1] == i:
                dist = np.linalg.norm(desc1[i] - desc2[match_1])
                if dist < self.threshold_matches:
                    mutual_matches[i] = (match_1, dist)

        return mutual_matches

    def ransac_translation(
        self,
        kpts_src,
        kpts_dst,
        desc_src,
        desc_dst,
        source_img,
        dest_image,
        vis_matches=False,
        vis_in_out=False,
    ):
        """
        Perform RANSAC to estimate the translation matrix between two sets of keypoints.
        :param kpts_src: Keypoints from the source image.
        :param kpts_dst: Keypoints from the destination image.
        :param desc_src: Descriptors from the source image.
        :param desc_dst: Descriptors from the destination image.
        :param source_img: Source image.
        :param dest_image: Destination image.
        :param vis_matches: Flag to visualize matches.
        :param vis_points: Flag to visualize points.
        :return: Best translation matrix and inliers.
        """
        # Compute mutual matches
        matches = self.mutual_match(desc_src, desc_dst)  # matches source to destination
        if vis_matches:
            visualize_matches(
                kpts_dst,
                kpts_src,
                matches,
                dest_image,
                source_img,
                self.threshold_matches,
            )
        self.matches = matches
        best_inliers = []
        best_residuals = []
        best_translation = None
        best_inliers_distances = []
        max_inliers = 0

        for _ in range(self.n_iterations):
            if self.sample_size >= 4:

                sample_indices = random.sample(list(matches.keys()), self.sample_size)
            else:
                KeyError(
                    f"Sample size {self.sample_size} is less than 4. RANSAC requires at least 4 points to compute a homography."
                )
            src_pts = np.array([kpts_src[i] for i in sample_indices])
            dst_pts = np.array([kpts_dst[matches[i][0]] for i in sample_indices])

            src_pts = np.array([keypoint.pt for keypoint in src_pts], dtype=np.float32)
            dst_pts = np.array([keypoint.pt for keypoint in dst_pts], dtype=np.float32)

            # affine_matrix = self.compute_translation_horizontal(src_pts, dst_pts)
            M, _ = cv2.findHomography(src_pts, dst_pts, 0)
            if M is None:
                continue
            all_src_indices = list(matches.keys())
            all_src_pts = np.array(
                [kpts_src[i].pt for i in all_src_indices], dtype=np.float32
            )

            all_src_pts_transformed = cv2.perspectiveTransform(
                np.array([all_src_pts]), M
            )[0]

            inliers = []
            distances_in = []
            residuals = []
            for idx, i in enumerate(all_src_indices):
                j = matches[i][0]
                target_pt = np.array(kpts_dst[j].pt)
                # print("estimate:\n", all_src_pts_transformed[idx])
                # print("target_pt:\n", target_pt)
                dist = np.linalg.norm(all_src_pts_transformed[idx] - target_pt)
                residuals.append(dist)
                # print("dist:\n", dist)
                if dist < self.threshold:
                    inliers.append(i)
                    distances_in.append(dist)

                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_inliers = inliers
                    best_inliers_distances = distances_in
                    best_translation = M
                    best_residuals = residuals
            # print(
            #     f"Iteration: {_}, Inliers: {len(inliers)}, Max Inliers: {max_inliers}"
            # )
        self.residuals = best_residuals
        self.translation_matrix = best_translation
        self.inliers = best_inliers
        self.outliers = list(set(all_src_indices) - set(best_inliers))
        self.average_residual_inliers = np.mean(best_inliers_distances)
        if vis_in_out:
            visualize_inliers_and_outliers(
                source_img,
                dest_image,
                best_inliers,
                self.outliers,
                kpts_src,
                kpts_dst,
                matches,
            )

        return best_translation, best_inliers

    def stitch_images(self, dest_image, source_img, affine_matrix, borders=False):
        """
        Stitch two images together using the estimated translation matrix.
        :param image1: Destination image.
        :param image2: Source image.
        :return: Stitched image.
        """
        panorama_height = dest_image.shape[0]
        panorama_width = dest_image.shape[1] + source_img.shape[1]
        panorama_size = (panorama_width, panorama_height)
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        panorama[: dest_image.shape[0], : dest_image.shape[1]] = dest_image

        img2_warped = cv2.warpPerspective(source_img, affine_matrix, panorama_size)
        if borders:
            h2, w2 = source_img.shape[:2]
            corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(
                corners, affine_matrix
            ).astype(np.int32)
            cv2.polylines(
                panorama,
                [transformed_corners],
                isClosed=True,
                color=(0, 255, 0),
                thickness=4,
            )
        mask = np.all(img2_warped > [0, 0, 0], axis=2)
        panorama[mask] = img2_warped[mask]

        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Image")
        plt.axis("off")
        plt.show()


def add_offset(kpts, offset):
    # Add the offset to the keypoints in the source image

    keypoints_r_offset = []
    for keypoint in kpts:
        new_keypoint = cv2.KeyPoint(
            x=keypoint.pt[0] + offset,
            y=keypoint.pt[1],
            size=keypoint.size,
            angle=keypoint.angle,
            response=keypoint.response,
            octave=keypoint.octave,
            class_id=keypoint.class_id,
        )

        keypoints_r_offset.append(new_keypoint)

    return keypoints_r_offset


def plot_matches_inliers_outliers(
    ax, kpts_src, kpts_dest, matches, inliers, outliers, image_src, image_dest
):
    """
    Plot the matches between two sets of keypoints, highlighting inliers and outliers.
    :param ax: Matplotlib axis to plot on.
    :param kpts_src: Keypoints from the source image.
    :param kpts_dest: Keypoints from the destination image.
    :param matches: Dictionary of matches.
    :param inliers: List of inlier indices.
    :param outliers: List of outlier indices.
    :param image_src: Source image.
    :param image_dest: Destination image.
    """

    h_left, _ = image_dest.shape[:2]
    h_right, _ = image_src.shape[:2]
    max_height = max(h_left, h_right)

    if h_left < max_height:
        pad = max_height - h_left
        image_dest = np.pad(image_dest, ((0, pad)), mode="constant")
    if h_right < max_height:
        pad = max_height - h_right
        image_src = np.pad(image_src, ((0, pad)), mode="constant")

    offset = image_dest.shape[1]
    kpts_src = add_offset(kpts_src, offset)
    combined_img = np.hstack((image_dest, image_src))
    ax.imshow(combined_img)
    ax.axis("off")
    (inlier_line,) = ax.plot([], [], "go", markersize=5)
    (outlier_line,) = ax.plot([], [], "ro", markersize=5)
    ax.plot([], [], "g-", label="Inliers")
    ax.plot([], [], "r-", label="Outliers")

    # Plot inliers
    for i in inliers:
        pt1 = kpts_src[i].pt
        pt2 = kpts_dest[matches[i][0]].pt
        x1, y1 = pt1
        x2, y2 = (
            pt2[0],
            pt2[1],
        )
        ax.plot([x1, x2], [y1, y2], "g-", linewidth=1.5)
        ax.plot(x1, y1, "go", markersize=5)
        ax.plot(x2, y2, "go", markersize=5)

    # Plot outliers
    for j in outliers:
        pt1 = kpts_src[j].pt
        pt2 = kpts_dest[matches[j][0]].pt
        x1, y1 = pt1
        x2, y2 = (
            pt2[0],
            pt2[1],
        )
        ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)
        ax.plot(x1, y1, "ro", markersize=5)
        ax.plot(x2, y2, "ro", markersize=5)

    ax.legend(
        [inlier_line, outlier_line],
        ["Inliers", "Outliers"],
        loc="upper left",
        fontsize=12,
    )


def visualize_inliers_and_outliers(
    image_src,
    image_dest,
    inliers,
    outliers,
    kpts_src,
    kpts_dest,
    matches,
    window_name="Inliers and Outliers",
):
    """
    Visualize inliers and outliers in the matched keypoints.
    :param image_src: Source image.
    :param image_dest: Destination image.
    :param inliers: List of inlier indices.
    :param outliers: List of outlier indices.
    :param kpts_src: Keypoints from the source image.
    :param kpts_dest: Keypoints from the destination image.
    :param matches: Dictionary of matches.
    :param window_name: Name of the window for visualization.
    """

    _, ax = plt.subplots(figsize=(15, 8))

    plot_matches_inliers_outliers(
        ax, kpts_src, kpts_dest, matches, inliers, outliers, image_src, image_dest
    )
    plt.title(f"Matched Keypoints, Inliers and Outliers", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_matches(kpts_l, kpts_r, matches, left_img, right_img, threshold):
    """
    Visualize the matches between two sets of keypoints.
    :param kpts_l: Keypoints from the left image.
    :param kpts_r: Keypoints from the right image.
    :param matches: Dictionary of matches.
    :param left_img: Left image.
    :param right_img: Right image.
    :param threshold: Distance threshold for considering a match as valid.
    """

    h_left, _ = left_img.shape[:2]
    h_right, _ = right_img.shape[:2]
    max_height = max(h_left, h_right)

    if h_left < max_height:
        pad = max_height - h_left
        left_img = np.pad(left_img, ((0, pad)), mode="constant")
    if h_right < max_height:
        pad = max_height - h_right
        right_img = np.pad(right_img, ((0, pad)), mode="constant")

    combined_img = np.hstack((left_img, right_img))
    offset = left_img.shape[1]
    kpts_r = add_offset(kpts_r, offset)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(combined_img)
    ax.axis("off")

    for j, (i, dist) in matches.items():
        if dist < threshold:
            pt1 = kpts_l[i].pt
            pt2 = kpts_r[j].pt

            x1, y1 = pt1
            x2, y2 = pt2[0], pt2[1]

            ax.plot([x1, x2], [y1, y2], "g-", linewidth=1.5)
            ax.plot(x1, y1, "ro", markersize=5)
            ax.plot(x2, y2, "ro", markersize=5)

    plt.title(f"Matched Keypoints (Threshold = {threshold})")
    plt.tight_layout()
    plt.show()
