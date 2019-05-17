import numpy as np
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles


class Distance(Metric):
    def __init__(self):
        super(Distance, self).__init__(feature=ResampleFeature(nb_points=30))

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    def dist(self, features1, features2):
        distance = 0
        for i in range(len(features1)):
            distance += np.linalg.norm(features1[i] - features2[i])
        return distance


def test_qb(dataset):
    metric = Distance()
    qb = QuickBundles(threshold=1, metric=metric)
    clusters = qb.cluster(dataset)
    print('clusters:', clusters)
