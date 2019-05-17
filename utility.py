import matplotlib.pyplot as plt
import numpy as np


def format_dataset(dataset):
    """
    数据预处理
    :param dataset:
    :return:
    """
    # TODO: 轨迹方向判断和反转

    # 字典格式 => 数组格式
    result = []
    for data in dataset:
        trajectory = []
        for point in data:
            trajectory.append([point['x'], point['y']])
        result.append(np.asarray(trajectory))

    return result


def format_result(result):
    """
    结果转化为可输出的格式
    :param result:
    :return:
    """
    new_result = []
    for point in result:
        new_result.append({
            'x': point[0],
            'y': point[1]
        })
    return new_result


def print_result_graph(dataset, result):
    """
    打印一组轨迹及其结果
    :param dataset: 原始数据
    :param result: 一条轨迹结果
    :return:
    """
    for trajectory in dataset:
        trajectory = np.asarray(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b', marker='o')
    result = np.asarray(result)
    plt.plot(result[:, 0], result[:, 1], 'r', marker='o')
    plt.show()