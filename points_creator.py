import random
import csv
import math
import argparse

def generate_random_point(original_point, r):
    n = len(original_point)
    angles = [random.uniform(0, 2 * math.pi) for _ in range(n)]
    distance = random.gauss(0, 1) * r

    offsets = []
    for i in range(n):
        if i == n - 1:
            coord = distance
            for angle in angles[:i]:
                coord *= math.cos(angle)
            offsets.append(coord)
        else:
            coord = distance * math.sin(angles[i])
            for angle in angles[:i]:
                coord *= math.cos(angle)
            offsets.append(coord)

    new_point = [original_point[i] + offsets[i] for i in range(n)]

    return new_point

def main(amount, dimensions, max_value, max_bias, clusters, file):
    with open('points.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for _ in range(clusters):
            bias = generate_random_point([0] * dimensions, max_bias)
            for _ in range(amount):
                writer.writerow(generate_random_point(bias, max_value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random points for clustering.')
    parser.add_argument('--amount', type=int, default=150, help='Number of points per cluster')
    parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--max_value', type=int, default=250, help='Max value for generating points')
    parser.add_argument('--max_bias', type=int, default=1500, help='Max bias for cluster center')
    parser.add_argument('--clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--file', type=str, default='points.csv', help='Name of the file')

    args = parser.parse_args()

    main(args.amount, args.dimensions, args.max_value, args.max_bias, args.clusters, args.file)
