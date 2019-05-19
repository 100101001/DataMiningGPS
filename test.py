import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import ResampleFeature
from utility import print_result_graph
# pip/pip3 install dipy


def try_resample(dataset):
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

    denoise_data, result = denoising(trans_set)

    # print_result_graph(denoise_data, result)

    return result


def denoising(dataset):
    new_dataset = []
    avg_trajectory = []

    # def is_outlier(suspect):
    #     avg_point_without_suspect = [(avg_point[k]*len(point_set) - suspect[k]) / (len(point_set)-1) for k in range(2)]
    #     distance = np.linalg.norm(avg_point - avg_point_without_suspect)
    #     return distance > 0.06

    def is_outlier(suspect):
        distance = np.linalg.norm(suspect - avg_point)
        return distance > (3 * sigma)

    for i in range(len(dataset[0])):
        point_set = []
        for j in range(len(dataset)):
            point_set.append(dataset[j][i])
        # 平均数点
        avg_point = np.mean(point_set, axis=0)
        # 标准差
        sigma = np.math.sqrt(sum([np.linalg.norm(point - avg_point)**2 for point in point_set]) / len(point_set))

        # 找到噪音点
        final_point_set = []
        for j in range(len(point_set)):
            if is_outlier(point_set[j]):
                dataset[j][i] = [-1, -1]
            else:
                final_point_set.append(dataset[j][i])

        final_avg_point = np.mean(final_point_set, axis=0) if (len(final_point_set) > 0) else avg_point
        avg_trajectory.append(final_avg_point)

    # 去掉噪音点
    for trajectory in dataset:
        new_trajectory = []
        for point in trajectory:
            if point[0] != -1:
                new_trajectory.append(point)
        if len(new_trajectory) >= 2:
            new_dataset.append(new_trajectory)

    return new_dataset, avg_trajectory






