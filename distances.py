import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


def normalize_zmuv(desc):
    desc_norm = desc - np.mean(desc, axis=1, keepdims=True)
    desc_norm /= np.linalg.norm(desc_norm, axis=1, keepdims=True) + 1e-8
    return desc_norm


def normalize_descriptor(desc):
    desc1_ncc = normalize_zmuv(desc)
    desc1_l2 = normalize(desc1_ncc, axis=1)
    return desc1_l2


def compute_euclidian_distances(desc1, desc2):

    desc1_l2 = normalize_descriptor(desc1)
    desc2_l2 = normalize_descriptor(desc2)
    euclidean_distance = cdist(desc1_l2, desc2_l2, metric="euclidean")
    return euclidean_distance


def get_matches(desc1, desc2, threshold=0.3):
    distances = compute_euclidian_distances(desc1, desc2)
    matches = {}

    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])
        if distances[i, sorted_indices[0]] < threshold:
            matches[i] = (
                sorted_indices[0],
                distances[i, sorted_indices[0]],
            )  # (index, distance)

    return matches


def plot_box_plots_for_distances(results):
    # Prepara i dati per il box plot
    distances_per_patch_size = [distances for _, distances in results.values()]

    # Crea il box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        distances_per_patch_size, labels=[f"Patch {size}" for size in results.keys()]
    )
    plt.title("Distribuzione delle distanze per ogni Patch Size")
    plt.xlabel("Dimensione Patch")
    plt.ylabel("Distanza delle Corrispondenze")
    plt.grid(True)
    plt.show()


def plot_box_plots_distances_threshold(results):
    distances_per_th = []
    for threshold, _ in results.items():
        distances = []

        for _, values in results[threshold].items():
            distances.append(values[1])

        distances_per_th.append(distances)
        print(
            "Threshold: ",
            threshold,
            "mean: ",
            np.mean(distances),
            "std: ",
            np.std(distances),
        )

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        distances_per_th,
        labels=[f"Threshold = {th}" for th in results.keys()],
        showmeans=True,
    )
    plt.title("Distance of Matches for Different Patch Sizes")
    plt.xlabel("Patch Size")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()


def plot_box_plots_distances_top_n(results):
    distances = []
    for len in results:
        value = results[len]
        dist = []

        for pair in value:
            dist.append(pair[1][1])
        distances.append(dist)
        print("n: ", len, "mean: ", np.mean(dist), "std: ", np.std(dist))

    plt.figure(figsize=(10, 6))
    plt.boxplot(distances, labels=[f"n = {n}" for n in results.keys()], showmeans=True)
    plt.title("Distance of Matches for Different n")
    plt.xlabel("n")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()
