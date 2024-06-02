import random
import math
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np

def generate_random_medians(points, clusters=10):
    cluster_medians = []
    for cluster in range(clusters):
        random_point = list(points[random.randint(0, len(points) - 1)])
        while list(random_point) in cluster_medians:
            random_point = list(points[random.randint(0, len(points) - 1)])
        cluster_medians.append(random_point)
    cluster_medians = np.array(cluster_medians)
    return cluster_medians

def calc_distance(a, b):
    distance = 0
    for val in range(len(a)):
        distance += (a[val] - b[val]) ** 2
    distance = math.sqrt(distance)
    return distance

def calc_clusters(points, cluster_medians):
    point_clusters = []
    for point in points:
        point_cluster = 0
        min_distance = calc_distance(point, cluster_medians[0])
        for i, median in enumerate(cluster_medians[1:]):
            new_min_distance = calc_distance(point, median)
            if new_min_distance < min_distance:
                min_distance = new_min_distance
                point_cluster = i + 1
        point_clusters.append(point_cluster)

    point_clusters = np.array(point_clusters)
    return point_clusters

def cluster_median(points, cluster_indices):
    unique_clusters = np.unique(cluster_indices)
    cluster_medians = []

    for cluster in unique_clusters:
        cluster_points = points[cluster_indices == cluster]
        median = np.median(cluster_points, axis=0)
        cluster_medians.append(median)

    return np.array(cluster_medians)

def cluster_variance(points, cluster_indices):
    unique_clusters = np.unique(cluster_indices)
    cluster_variances = []

    for cluster in unique_clusters:
        cluster_points = points[cluster_indices == cluster]
        variance = np.var(cluster_points)
        cluster_variances.append(variance * len(cluster_points))
    return np.array(cluster_variances)

def calc_best(points, clusters, cycles=10):
    current_best = math.inf
    best_point_clusters = []
    best_cluster_medians = []
    for _ in range(cycles):
        cluster_medians = generate_random_medians(points, clusters)
        point_clusters = calc_clusters(points, cluster_medians)
        current_variance = sum(cluster_variance(points, point_clusters))
        previous_variance = current_variance + 1
        while previous_variance > current_variance:
            cluster_medians = cluster_median(points, point_clusters)
            point_clusters = calc_clusters(points, cluster_medians)
            previous_variance = current_variance
            current_variance = sum(cluster_variance(points, point_clusters))
        if current_variance < current_best:
            best_point_clusters = point_clusters
            best_cluster_medians = cluster_medians
            current_best = current_variance
    return best_point_clusters, best_cluster_medians, current_best

def main(file, result_file, clusters, tries):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        points = list(spamreader)
        points = np.array([[float(x), float(y)] for x, y in points])

    point_clusters, cluster_medians, current_best = calc_best(points, clusters, tries)

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    median_x_coords = [median[0] for median in cluster_medians]
    median_y_coords = [median[1] for median in cluster_medians]

    colors = plt.get_cmap(lut=len(cluster_medians))

    cluster_colors = [colors(cluster) for cluster in point_clusters]

    print(cluster_medians)

    plt.scatter(x_coords, y_coords, c=cluster_colors, s=100)
    plt.scatter(median_x_coords, median_y_coords, c='red', s=200, marker='X', label='Cluster Medians')  # Larger size and different marker
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Scatter Plot of Points')
    plt.grid(True)
    plt.savefig(result_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Elbow method for K-medians')
    parser.add_argument('--file', type=str, default='points.csv', help='Path to the input CSV file containing points')
    parser.add_argument('--result_file', type=str, default='K-medians.png', help='Path to save the resulting plot image')
    parser.add_argument('--clusters', type=int, default=10, help='Number of clusters to use')
    parser.add_argument('--tries', type=int, default=100, help='Number of attempts for each k value')

    args = parser.parse_args()

    main(args.file, args.result_file, args.clusters, args.tries)
