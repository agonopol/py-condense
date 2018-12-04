import numpy as np, math
from pandas.core.frame import DataFrame
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from sklearn.neighbors import NearestNeighbors


def affinity(data, epsilon):
    distances = squareform(pdist(data, metric='sqeuclidean'))
    return exp(-distances / epsilon ** 2)


class Condensator:
    def __init__(self, data, **kwargs):
        self.data = data
        if isinstance(data, DataFrame):
            self.data = data.values
        if 'n' in kwargs:
            self.n = kwargs['n']
        else:
            self.n = 2
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        else:
            self.sigma = 2.2204e-16
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            knn = NearestNeighbors().fit(self.data)
            distances, _ = knn.kneighbors(self.data)
            self.epsilon = np.percentile(distances[:, 4], .95)
        if 'affinity' in kwargs:
            self.affinityfn = kwargs['affinity']
        else:
            self.affinityfn = affinity

    def contract(self):
        if len(self.data) > self.n:
            assigments, condesator = self.next( )
        while len(condesator.data) > self.n:
            yield assigments, condesator.epsilon
            assigments, condesator = condesator.next( )

    def next(self):
        diffused = self.diffuse()
        distances = squareform(pdist(diffused, metric='sqeuclidean'))
        if np.any(distances < self.sigma - np.eye(len(distances), len(distances))):
            [merged, data] = self.merge(diffused, distances)
            return merged, Condensator(data, sigma=self.sigma + math.e, n=self.n, epsilon=self.epsilon,
                                       affinity=self.affinityfn)
        else:
            return [], Condensator(diffused, sigma=self.sigma, n=self.n, epsilon=self.epsilon * 1.05,
                                   affinity=self.affinityfn)

    def diffuse(self):
        affinity = self.affinity()
        data = ((affinity / affinity.sum(1)) ** 2) @ self.data
        return data

    def merge(self, diffused, distances):
        components = self.components( distances )

    def components(self, distances):
        filtered = (distances < self.sigma ** 2) - np.eye(len(distances))
        sums = np.sum(filtered, axis=1)
        over = sums > 0
        under = sums <= 0

    def laplacian(self, distances):
        filtered = (distances < self.sigma ** 2) - np.eye(len(distances))


    def affinity(self):
        return self.affinityfn(self.data, self.epsilon)


if __name__ == '__main__':
    import pandas
    import os

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data", "circles.csv")) as circles:
        data = pandas.read_csv(circles)
        condesator = Condensator(data)
        for i, assigment in enumerate(condesator.contract()):
            print( i, assigment )

