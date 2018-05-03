from scipy.stats import multivariate_normal
from scipy.stats import laplace
from matplotlib import pyplot as plt
import json
from itertools import groupby
from tqdm import tqdm

import numpy as np

init = 1

def metric(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

# def metric(x, y):
#     s = 0
#     for index in range(len(x)):
#         if x[index] != y[index]: s+=1
#     return s*100

def calculate_identifiers(points, weights, means):
    probabilities = np.array([np.array([laplace.pdf(metric(points[i], means[j]))  
                                        for j in range(0, means.shape[0])])
                              for i in range(0, len(points))])

    return np.array([np.array([weights[j] * probabilities[i][j] / np.sum(probabilities[i].dot(weights))
                               for j in range(0, means.shape[0])]).argmax()
                     for i in range(0, len(points))])

def calculate_parameters(points, identifiers, class_count):
    sums = np.zeros((class_count, 1))
    for i in identifiers:
        sums[i] += 1
    weights = sums / len(points)

    means = np.zeros((class_count, len(points[0])))
    for i in range(0, len(points)):
        means[identifiers[i]] += points[i]
    means = means / sums

    return weights, means

def find_classes(points, class_count, iterations = 20):
    means = np.copy(points[:class_count])
    variances = np.full((class_count, len(points[0])), init)
    weights = np.full((class_count), 1 / class_count)

    for i in tqdm(range(iterations)):
        identifiers = calculate_identifiers(points, weights, means)
        weights, means = calculate_parameters(points, identifiers, class_count)

    return points, identifiers

def read_values(filename):
    _data = None
    _types = None
    with open(filename) as f:

        data = json.load(f)
        f.close

        classes = data['classes']

        for index in range(len(classes)):
            for dot in data[classes[index]]:
                if _data is None:
                    _data = np.array(dot)
                    _types = np.array(index)
                else:
                    _data = np.vstack([_data, dot])
                    _types = np.vstack([_types, index])
    
    return _data, _types

def testing():
    data, types = read_values('./sample (done)/sample.json')
    plt.figure(0)
    plt.title('Initial')
    plt.scatter(data[:, 0], data[:, 1])
    # print(data)
    number_of = 5
    data, types = find_classes(data, number_of)
    plt.figure(1)
    plt.title('Generated')
    plt.scatter(data[:, 0], data[:, 1], c=types)
    plt.show()

if __name__ == '__main__':
    testing()