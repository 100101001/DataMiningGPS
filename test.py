import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import ResampleFeature
# pip/pip3 install dipy


def try_resemble(dataset):
    res = ResampleFeature(nb_points=30)
    for trajectory in dataset:
        trajectory = np.asarray([np.asarray(p) for p in trajectory])
        trans = res.extract(trajectory)
        plt.plot(trans[:, 0], trans[:, 1], 'b', marker='o')
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r', marker='o')
        plt.show()
        print()


def try_avg(dataset, points_count):
    # 重新取样
    resample = ResampleFeature(nb_points=points_count)
    trans_set = []
    for trajectory in dataset:
        trajectory = np.asarray([np.asarray(p) for p in trajectory])
        trans_trajectory = resample.extract(trajectory)
        trans_set.append(trans_trajectory)

    # 取平均数
    result = []
    for i in range(points_count):
        result.append([0, 0])
        for j in range(len(trans_set)):
            result[i][0] += trans_set[j][i][0]
            result[i][1] += trans_set[j][i][1]
    for i in range(points_count):
        result[i][0] /= len(trans_set)
        result[i][1] /= len(trans_set)

    return result

