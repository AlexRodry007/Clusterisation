import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
import numpy as np


def main(file, result_file, max_iterations):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        points = list(spamreader)
        points = np.array([[float(x), float(y)] for x, y in points])

    S = -pairwise_distances(points, metric='sqeuclidean')

    ap = AffinityPropagation(affinity='precomputed', max_iter=10000)
    ap.fit(S)

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    center_x_coords = [points[center_index][0] for center_index in cluster_centers_indices]
    center_y_coords = [points[center_index][1] for center_index in cluster_centers_indices]

    if len(np.unique(labels)) <= 10:
        colors = plt.get_cmap('tab10', lut=len(np.unique(labels)))
    else:
        colors = plt.get_cmap(lut=len(np.unique(labels)))

    cluster_colors = [colors(cluster) for cluster in labels]

    plt.scatter(x_coords, y_coords, c=cluster_colors, s=10)
    plt.scatter(center_x_coords, center_y_coords, c='red', s=50, marker='X', label='Cluster Medians')  # Larger size and different marker
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Scatter Plot of Points')
    plt.savefig(result_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mean-shift clustering algorithm with animation.')
    parser.add_argument('--file', type=str, default='points.csv', help='Path to the input CSV file containing points')
    parser.add_argument('--result_file', type=str, default='affinity_propagaion.png', help='Path to save the resulting plot image')
    parser.add_argument('--max_iterations', type=int, default=10000, help='Amount of maximum iteration of fitting')
    args = parser.parse_args()

    main(args.file, args.result_file, args.max_iterations)

