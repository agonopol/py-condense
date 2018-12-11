import numpy as np, math
from pandas.core.frame import DataFrame
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

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
            self.sigma = np.spacing(1)
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            knn = NearestNeighbors().fit(self.data)
            distances, _ = knn.kneighbors(self.data)
            self.epsilon = np.percentile(distances[:, 4], 95, interpolation='midpoint')
        if 'affinity' in kwargs:
            self.affinityfn = kwargs['affinity']
        else:
            self.affinityfn = affinity
        if 'i' in kwargs:
            self.i = kwargs['i']
        else:
            self.i = 1

    def contract(self):
        condesator = self
        while len(condesator.data) > self.n:
            assigments, condesator = condesator.next( )
            yield assigments, condesator.epsilon

    def next(self):
        diffused = self.diffuse()
        [merged, condensed] = self.merge( diffused )
        if merged:
            return merged, Condensator( condensed, sigma=self.sigma + math.e, n=self.n, epsilon=self.epsilon,
                                       affinity=self.affinityfn, i=self.i + 1)
        else:
            epsilon = self.epsilon * 1.05 if self.i % 200 == 0 else self.epsilon
            return [], Condensator( diffused, sigma=self.sigma, n=self.n, epsilon=epsilon,
                                   affinity=self.affinityfn, i=1)

    def diffuse(self):
        affinity = self.affinity()
        norm = (affinity / affinity.sum(1)).transpose()
        data = np.linalg.matrix_power(norm, 2) @ self.data
        return data

    def merge(self, diffused ):
        distances = squareform(pdist(diffused, metric='sqeuclidean'))
        filtered = (distances < self.sigma ** 2) - np.eye(len(distances))
        n, labels = connected_components(filtered)
        if n == len(filtered):
            return [], diffused
        else:
            merged = []
            data = np.zeros([n, diffused.shape[1]])
            for label in np.unique(labels):
                merge = label == labels
                if sum(merge) > 1:
                    merged.append(np.where(merge)[0])
                data[label] = np.average(diffused[labels == label])
            return merged, data



    def affinity(self):
        return self.affinityfn(self.data, self.epsilon)


if __name__ == '__main__':
    import scipy.io as sio
    import os
    sample = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data", "sample.mat"))
    condesator = Condensator(sample['pc'])
    for i, assigment in enumerate(condesator.contract()):
        print( i, assigment )

