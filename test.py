import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import ResampleFeature
import matplotlib.mlab as mlab
from deletePoints import delete_point_all
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
        # plt.figure(figsize=(6, 3))
        # plt.subplot(1, 2, 1)
        # utility.print_dataset(dataset)

        # 轨迹变硬
        # dataset = delete_point_all(dataset)
        #
        # dataset = utility.resample(dataset, 10)
        # plt.subplot(1, 2, 2)
        # utility.print_dataset(dataset)
        # plt.show()
        #
        # dataset = denoising(dataset)

        # 删除短轨迹
        # if len(dataset) > 2:
        #     dataset = delete_short_trajectory(dataset)
        #
        # plt.subplot(3, 1, 2)
        # utility.print_dataset(dataset)

        # 加点
        # dataset = process_short_trajectory(dataset)
        # plt.subplot(3, 1, 1)
        # utility.print_dataset(dataset)

        # 重新取样
        # for m in range(len(dataset)):
        #     for j in range(len(dataset[m])):
        #         dataset[m][j] = np.asarray(dataset[m][j])
        #     dataset[m] = np.asarray(dataset[m])
        dataset = utility.resample(dataset, points_count)

        # 去噪
        # dataset = denoising(dataset)

        # 取平均
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
    if len(dataset) <= 2:
        return dataset

    new_dataset = []
    # avg_trajectory = []

    def is_outlier(suspect_index):
        """ sigma变化法 """
    #     suspect = point_set.pop(suspect_index)
    #     cur_avg_point = np.mean(point_set, axis=0)
    #     cur_sigma = np.math.sqrt(sum([np.linalg.norm(p - cur_avg_point) ** 2 for p in point_set]) / len(point_set))
    #     point_set.insert(suspect_index, suspect)
    #     return cur_sigma < sigma * 0.6

        """ avg变化法 """
        # suspect = point_set[suspect_index]
        # avg_point_without_suspect = [(avg_point[k]*len(point_set) - suspect[k]) / (len(point_set)-1) for k in range(2)]
        # distance = np.linalg.norm(avg_point - avg_point_without_suspect)
        # return distance > 0.06

        """ k sigma 法 """
        suspect = point_set[suspect_index]
        distance = np.linalg.norm(suspect - avg_point)
        return distance > (2 * sigma)

    for i in range(len(dataset[0])):
        point_set = []
        for j in range(len(dataset)):
            point_set.append(dataset[j][i])
        if len(point_set) <= 1:
            continue
        # 平均数点
        avg_point = np.mean(point_set, axis=0)
        # 标准差
        sigma = np.math.sqrt(sum([np.linalg.norm(point - avg_point)**2 for point in point_set]) / len(point_set))

        # 找到噪音点
        final_point_set = []
        for j in range(len(point_set)):
            if is_outlier(j):
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


def delete_short_trajectory(dataset):
    new_set = []
    length = []
    for trajectory in dataset:
        l = np.linalg.norm(np.asarray(trajectory[0]) - np.asarray(trajectory[-1]))
        if l > 0.4:
            new_set.append(trajectory)
        else:
            print('delete:', l)
    return new_set


def process_short_trajectory(dataset):
    """
    处理长度太短的轨迹
    :param dataset: 原始数据集
    :return: 新数据集
    """
    # get avg points
    first_points = []
    last_points = []
    avg_first_point = []
    avg_last_point = []
    for trajectory in dataset:
        if np.linalg.norm(np.asarray(trajectory[0]) - np.asarray(trajectory[-1])) > 0.5:
            first_points.append(trajectory[0])
            last_points.append(trajectory[-1])
    if len(first_points) > 0:
        avg_first_point = np.mean(first_points, axis=0).reshape(1, 2)
    if len(last_points) > 0:
        avg_last_point = np.mean(last_points, axis=0).reshape(1, 2)

    for i, trajectory in enumerate(dataset):
        if np.linalg.norm(np.asarray(trajectory[0]) - np.asarray(trajectory[-1])) <= 0.5:
            trajectory = np.insert(trajectory, 0, avg_first_point, axis=0)
            trajectory = np.append(trajectory, avg_last_point, axis=0)
            dataset[i] = trajectory

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


def show_sigma_distribution(dataset):
    for i in range(len(dataset[0])):
        sigmas = []
        point_set = []
        for j in range(len(dataset)):
            point_set.append(dataset[j][i])
        # 平均数点
        avg_point = np.mean(point_set, axis=0)
        # 标准差
        sigma = np.math.sqrt(sum([np.linalg.norm(point - avg_point)**2 for point in point_set]) / len(point_set))

        for j in range(len(dataset)):
            current = point_set[0]
            point_set.pop(0)
            cur_avg_point = np.mean(point_set, axis=0)
            cur_sigma = np.math.sqrt(sum([np.linalg.norm(p - cur_avg_point)**2 for p in point_set]) / len(point_set))
            sigmas.append(cur_sigma)
            point_set.append(current)

        n, bins, patches = plt.hist(sigmas, 100, density=True, facecolor='g', alpha=0.75)
        y = mlab.normpdf(bins, np.mean(sigmas), sigma)
        plt.plot(bins, y, 'r--')
        # print(sigmas)
        print('avg:', np.mean(sigmas), 'sigma', sigma)
    plt.show()


