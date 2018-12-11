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
        condesator = self
        while len(condesator.data) > self.n:
            assigments, condesator = condesator.next( )
            yield assigments, condesator.epsilon

    def next(self):
        diffused = self.diffuse()
        distances = squareform(pdist(diffused, metric='sqeuclidean'))
        if np.any(distances < self.sigma - np.eye(len(distances), len(distances))):
            [merge, condesed] = self.merge(diffused, distances)
            return merge, Condensator(condesed, sigma=self.sigma + math.e, n=self.n, epsilon=self.epsilon,
                                       affinity=self.affinityfn)
        else:
            return [], Condensator(diffused, sigma=self.sigma, n=self.n, epsilon=self.epsilon * 1.005,
                                   affinity=self.affinityfn)

    def diffuse(self):
        affinity = self.affinity()
        data = ((affinity / affinity.sum(1)) ** 2) @ self.data
        return data

    def merge(self, diffused, distances):
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
    import pandas
    import os
    # Predefine choice
    choice = [279, 899, 727,  11, 195, 701, 413, 389, 371, 858, 300, 668,  94,
        36, 487, 279, 219, 976, 670, 873, 206, 786,  71, 667,  19, 134,
       524, 513, 417, 682, 860, 482, 481,   0, 148,  37, 469, 521, 768,
       602, 996, 841, 650,  22, 675, 390, 603, 361,  25, 305, 723, 582,
       862, 917, 538, 967, 908, 633, 501, 854, 807, 453, 333, 572, 536,
         2, 771, 139, 363, 620, 250, 639, 722, 904, 882, 446, 875, 765,
       509, 838, 199, 356, 126, 531, 752, 881, 143, 818,  72, 805, 338,
       642, 166,  88, 200, 318,  75, 258, 732, 769]


    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data", "circles.csv")) as circles:
        data = pandas.read_csv(circles)
        condesator = Condensator(data.values[choice])
        for i, assigment in enumerate(condesator.contract()):
            print( i, assigment )

