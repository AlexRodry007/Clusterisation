import math
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def calc_distance(a,b):
    distance = 0
    for val in range(len(a)):
        distance+=(a[val]-b[val])**2
    distance = math.sqrt(distance)
    return distance

def mean_shift_cycle(points, gravity_range, gravity_power):
    table = np.array(np.zeros((len(points), len(points))))
    for col, first_point in enumerate(points):
        for row, second_point in enumerate(points[col+1:]):
            row = row+col+1
            table[row, col] = calc_distance(first_point, second_point)
            table[col, row] = table[row, col]
    in_radius_table = table < gravity_range
    gravity_force = gravity_power

    movement_table = np.array(np.zeros((len(points), len(points[0]))))

    for row, row_value in enumerate(table):
        in_radius_points = []
        for column, value in enumerate(row_value):
            if in_radius_table[row][column]:
                in_radius_points.append(points[column])
        movement_table[row] = (np.mean(in_radius_points, axis=0) - points[row])*gravity_force
    new_points = points + movement_table
    return new_points

def main(file, result_file, animation_file, gravity_range, gravity_power, max_iterations, movement_threshold, include_animation):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        points = list(spamreader)
        points = np.array([[float(x), float(y)] for x, y in points])
    global starting_points  
    starting_points = points

    iterations = 0
    for _ in range(max_iterations):
        next_points = mean_shift_cycle(starting_points, gravity_range, gravity_power)
        if np.sum(np.abs(next_points-starting_points)/len(starting_points)) < movement_threshold:
            break
        starting_points = next_points
        iterations+=1

    table = np.array(np.full((len(starting_points), len(starting_points)),gravity_range+1.0))
    for col, first_point in enumerate(starting_points):
        for row, second_point in enumerate(starting_points[col+1:]):
            row = row+col+1
            table[row, col] = calc_distance(first_point, second_point)
            table[col, row] = table[row, col]
    in_radius_table = table < gravity_range

    sub_table = table
    clusters = []
    for row in range(len(sub_table)):
        if not np.isinf(sub_table[row][0]):
            in_radius_table = sub_table < gravity_range
            point_cluster = []
            for column, value in enumerate(sub_table[row]):
                if in_radius_table[row][column]:
                    point_cluster.append(column)
                    sub_table[column] = np.array([np.inf]*len(sub_table[row]))
            clusters.append(point_cluster)

    flat_clusters = np.zeros((len(points)),dtype=int)
    for cluster_index, cluster in enumerate(clusters):
        for point_index in cluster:
            flat_clusters[point_index] = cluster_index

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]


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

    if include_animation==1:
        starting_points = points

        x_coords = [point[0] for point in starting_points]
        y_coords = [point[1] for point in starting_points]

        fig, ax = plt.subplots()
        scat = ax.scatter(x_coords, y_coords, s=25)
        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')
        ax.set_title('Scatter Plot of Points')
        ax.grid(True)

        def update(frame):
            global starting_points
            starting_points = mean_shift_cycle(starting_points, gravity_range, gravity_power)
            x_coords = [point[0] for point in starting_points]
            y_coords = [point[1] for point in starting_points]
            scat.set_offsets(list(zip(x_coords, y_coords)))
            return scat,

        anim = FuncAnimation(fig, update, frames=iterations, interval=200, blit=True)

        anim.save(animation_file, writer=PillowWriter(fps=10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mean-shift clustering algorithm with animation.')
    parser.add_argument('--file', type=str, default='points.csv', help='Path to the input CSV file containing points')
    parser.add_argument('--result_file', type=str, default='mean-shift_clustering.png', help='Path to save the resulting plot image')
    parser.add_argument('--animation_file', type=str, default='scatter_animation.gif', help='Path to save the animation GIF')
    parser.add_argument('--gravity_range', type=float, default=350, help='Range of gravity for mean shift')
    parser.add_argument('--gravity_power', type=float, default=(1/250)*25, help='Power of gravity for mean shift')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of iterations for mean shift')
    parser.add_argument('--movement_threshold', type=float, default=0.1, help='Movement threshold to stop the iteration')
    parser.add_argument('--with_animation', type=int, default=1, help='Generate animation or not')

    args = parser.parse_args()

    main(args.file, args.result_file, args.animation_file, args.gravity_range, args.gravity_power, args.max_iterations, args.movement_threshold, args.with_animation)

