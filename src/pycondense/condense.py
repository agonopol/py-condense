import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from pycondense.kernels import gaussian


class Condensator:
    def __init__(self, data, kernel=gaussian, k=2, t=2, sigma=np.spacing(1e3),  **kwargs):
        self.data = data
        self.k = k
        self.kernel = kernel
        self.sigma = sigma
        self.t = t
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            knn = NearestNeighbors().fit(self.data)
            distances, _ = knn.kneighbors(self.data)
            self.epsilon = np.percentile(distances[:, 4], 95, interpolation='midpoint')
        if 'weights' in kwargs:
            self.weights = kwargs['weights']
        else:
            self.weights = np.ones(self.data.shape[0])
        if 'i' in kwargs:
            self.i = kwargs['i']
        else:
            self.i = 1

    def cluster(self):
        assigments = dict([(k, {k}) for k in range(self.data.shape[0])])
        centeroids = self.data
        for assigments, centeroids in self.iter():
            pass
        return assigments, centeroids

    def iter(self):
        idx, generator = dict([(k, {k}) for k in range(self.data.shape[0])]), self
        while len(generator.data) > self.k:
            assigments, generator = generator.next(idx)
            if assigments:
                idx = dict([(k, set([x for s in [idx[i] for i in v] for x in s])) for k, v in assigments.items()])
            yield idx, generator.data

    def next(self, idx):
        [merged, condensed, weights] = self.merge(self.diffuse(idx=idx))
        return merged, Condensator(condensed, sigma=self.sigma, n=self.k,
                                   epsilon=self.epsilon * 1.05 if self.i % 10 == 0 and not merged else self.epsilon,
                                   weights=weights, kernel=self.kernel, i=self.i + 1 if not merged else 1)

    def diffuse(self, **kwargs) -> np.array:
        affinity = self.affinity(**kwargs)
        norm = (affinity / affinity.sum(1)).transpose()
        data = np.linalg.matrix_power(norm, self.t) @ self.data
        return data

    def merge(self, diffused):
        distances = squareform(pdist(diffused, metric='sqeuclidean'))
        filtered = (distances < self.sigma) - np.eye(len(distances))
        n, labels = connected_components(filtered)
        if n == len(filtered):
            return {}, diffused, np.ones(diffused.shape[0])
        else:
            merged = {}
            data = np.zeros([n, diffused.shape[1]])
            weights = np.zeros(n)
            for label in np.unique(labels):
                merged[label] = set(np.where(labels == label)[0].tolist())
                data[label] = np.average(diffused[labels == label], axis=0, weights=self.weights[labels == label])
                weights[label] = sum(self.weights[labels == label])
            return merged, data, weights

    def affinity(self, **kwargs):
        return self.kernel(self.data, self.epsilon, **kwargs)
