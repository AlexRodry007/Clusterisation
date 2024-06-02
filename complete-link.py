import math
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics

def calc_distance(a,b):
    distance = 0
    for val in range(len(a)):
        distance+=(a[val]-b[val])**2
    distance = math.sqrt(distance)
    return distance

def calc_new_table(table, min_value_row, min_value_col):
    new_table = np.array(np.ones(tuple(x-1 for x in table.shape))*np.inf)
    bias_r = 0
    for row, row_value in enumerate(table):
        bias_c = 0
        if row == min_value_row or row == min_value_col:
            bias_r+=1
        else: 
            for col, value in enumerate(row_value):
                if col == min_value_row or col == min_value_col:
                    bias_c+=1
                else:
                    new_table[row-bias_r, col-bias_c] = value
            new_table[row-bias_r][-1] = max(table[row][min_value_row],table[row][min_value_col])
    new_table[-1] = new_table.transpose()[-1]
    return new_table

def complete_linkage_cycle(table, clusters):
    min_value_row, min_value_col = np.unravel_index(np.argmin(table), table.shape)
    new_table = calc_new_table(table, min_value_row, min_value_col)
    clusters.append(clusters[min_value_col]+clusters[min_value_row])
    clusters.pop(max(min_value_col, min_value_row))
    clusters.pop(min(min_value_col, min_value_row))
    return new_table, clusters

def main(file, result_file):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        points = list(spamreader)
        points = np.array([[float(x), float(y)] for x, y in points])

    table = np.array(np.ones((len(points), len(points)))*np.inf)
    for col, first_point in enumerate(points):
        for row, second_point in enumerate(points[col+1:]):
            row = row+col+1
            table[row, col] = calc_distance(first_point, second_point)
            table[col, row] = table[row, col]

    clusters = [[i] for i in range(table.shape[0])]

    current_best_score = -math.inf
    best_flat_clusters = None

    while len(clusters) > 2:
        table, clusters = complete_linkage_cycle(table, clusters)
        flat_clusters = np.zeros((len(points)),dtype=int)
        for cluster_index, cluster in enumerate(clusters):
            for point_index in cluster:
                flat_clusters[point_index] = cluster_index
        new_score = sklearn.metrics.silhouette_score(points, flat_clusters)
        if new_score>current_best_score:
            best_flat_clusters = flat_clusters.copy()
            current_best_score = new_score

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    flat_clusters = best_flat_clusters

    if len(np.unique(flat_clusters)) <= 10:
        colors = plt.get_cmap('tab10', lut=len(np.unique(flat_clusters)))
    else:
        colors = plt.get_cmap(lut=len(np.unique(flat_clusters)))

    cluster_colors = [colors(cluster) for cluster in flat_clusters]

    plt.scatter(x_coords, y_coords, c=cluster_colors, s=10)
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Scatter Plot of Points')
    plt.grid(True)
    plt.savefig(result_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete-link method')

    parser.add_argument('--file', type=str, default='points.csv', help='Path to the input CSV file containing points')
    parser.add_argument('--result_file', type=str, default='complete-link.png', help='Path to save the resulting plot image')

    args = parser.parse_args()
    main(args.file, args.result_file)