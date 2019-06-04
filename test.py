import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import ResampleFeature
import utility
# pip/pip3 install dipy


def try_avg(dataset, points_count):
    """
    平均数方法
    :param dataset: 数据集
    :param points_count: 取样点数
    :return:
    """
    avg_tra = []

    for i in range(1):
        plt.figure(figsize=(3, 9))
        # 加点
        # dataset = process_short_trajectory(dataset)
        # plt.subplot(3, 1, 1)
        # utility.print_dataset(dataset)

        # 重新取样
        dataset = utility.resample(dataset, points_count)
        # plt.subplot(3, 1, 2)
        # utility.print_dataset(dataset)

        # 去噪
        dataset = denoising(dataset)
        avg_tra = get_avg_trajectory(dataset, points_count)
        # plt.subplot(3, 1, 3)
        # utility.print_result_graph(dataset, avg_tra)
        # plt.show()

    return avg_tra


def denoising(dataset):
    """
    去噪算法
    :param dataset: 原始数据集
    :return: 新数据集
    """
    new_dataset = []
    # avg_trajectory = []

    def is_outlier(suspect):
        avg_point_without_suspect = [(avg_point[k]*len(point_set) - suspect[k]) / (len(point_set)-1) for k in range(2)]
        distance = np.linalg.norm(avg_point - avg_point_without_suspect)
        return distance > 0.06

    # def is_outlier(suspect):
    #     distance = np.linalg.norm(suspect - avg_point)
    #     return distance > (3 * sigma)

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

        # final_avg_point = np.mean(final_point_set, axis=0) if (len(final_point_set) > 0) else avg_point
        # avg_trajectory.append(final_avg_point)

    # 去掉噪音点
    for trajectory in dataset:
        new_trajectory = []
        for point in trajectory:
            if point[0] != -1:
                new_trajectory.append(point)
        if len(new_trajectory) >= 2:
            new_dataset.append(new_trajectory)

    # plt.subplot(2, 1, 2)
    # utility.print_dataset(new_dataset)
    return new_dataset  # , avg_trajectory


def process_short_trajectory(dataset):
    """
    处理长度太短的轨迹
    :param dataset: 原始数据集
    :return: 新数据集
    """
    length = []
    short_traj = []
    first_points = []
    last_points = []
    for trajectory in dataset:
        length.append(np.linalg.norm(trajectory[0] - trajectory[1]))
    # 计算sigma
    avg_length = np.mean(length)
    sigma = np.math.sqrt(sum([(avg_length - l)**2 for l in length]) / len(length))
    # 计算平均点
    for i, l in enumerate(length):
        if avg_length > l + sigma:
            short_traj.append(i)
        else:
            first_points.append(dataset[i][0])
            last_points.append(dataset[i][-1])
    if len(first_points) > 0:
        avg_first_point = np.mean(first_points, axis=0).reshape(1, 2)
    if len(last_points) > 0:
        avg_last_point = np.mean(last_points, axis=0).reshape(1, 2)

    for i in short_traj:
        if len(first_points) > 0:
            dataset[i] = np.insert(dataset[i], 0, avg_first_point, axis=0)
        if len(last_points) > 0:
            dataset[i] = np.append(dataset[i], avg_last_point, axis=0)

    return dataset


def get_avg_trajectory(dataset, points_count):
    """
    获取一组轨迹的平均轨迹
    :param dataset: 数据集
    :param points_count: 重新取样的点数
    :return: 一条平均轨迹
    """
    avg_trajectory = []
    dataset = utility.resample(dataset, points_count)
    for i in range(len(dataset[0])):
        point_set = []
        for j in range(len(dataset)):
            point_set.append(dataset[j][i])
        # 平均数点
        avg_point = np.mean(point_set, axis=0)
        avg_trajectory.append(avg_point)
    return avg_trajectory

