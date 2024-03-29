import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import ResampleFeature


def format_dataset(dataset):
    """
    数据预处理
    :param dataset: 一份测试数据集，包括多条轨迹
    :return:
    """
    tmp = dataset[0][0]['x']
    tmp2 = dataset[0][0]['y']

    if abs(dataset[0][0]['x'] - dataset[0][-1]['x']) > 0.5:
        use_x = True
    else:
        use_x = True if abs(dataset[0][0]['x'] - dataset[0][-1]['x']) >= abs(dataset[0][0]['y'] - dataset[0][-1]['y']) else False

    result = []
    for data in dataset:
        trajectory = []
        for point in data:
            # 字典格式 => 数组格式
            trajectory.append([point['x'], point['y']])
        # print(trajectory)
        # 轨迹方向判断和反转
        if use_x:
            if data[0]['x'] > data[-1]['x']:
                trajectory.reverse()
        else:
            if data[0]['y'] > data[-1]['y']:
                trajectory.reverse()
        # if abs(data[0]['x'] - data[-1]['x']) < 0.3:
        #     if data[0]['y'] <= data[-1]['y'] != to_up:
        #         trajectory.reverse()
        # elif data[0]['x'] > data[-1]['x']:
        #     trajectory.reverse()

        # if (abs(data[0]['x'] - tmp) > 0.5)or(abs(data[0]['y'] - tmp2) > 0.5):
        #     trajectory.reverse()
        # print(trajectory)
        # print("---------")
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


"""
    plt.plot 的颜色表
    
    ‘b’	blue
    ‘g’	green
    ‘r’	red
    ‘c’	cyan
    ‘m’	magenta
    ‘y’	yellow
    ‘k’	black
    ‘w’	white
"""


def print_result_graph(dataset, result):
    """
    打印一组轨迹及其结果
    :param dataset: 原始数据
    :param result: 一条轨迹结果
    :return:
    """
    print_dataset(dataset, 'b')
    print_trajectory(result, 'r')


def print_dataset(dataset, color='b'):
    """
    打印一组轨迹
    :param dataset: 数据集
    :param color: 轨迹颜色
    :return:
    """
    for trajectory in dataset:
        trajectory = np.asarray(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], color, marker='o')


def print_trajectory(trajectory, color):
    """
    打印一条轨迹
    :param trajectory: 轨迹数据
    :param color: 轨迹颜色
    :return:
    """
    trajectory = np.asarray(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], color, marker='o')


def resample(dataset, points_count):
    """
    重新取样
    :param dataset: 数据集，含一条路上的多条轨迹
    :param points_count: 重新取样的点数
    :return:
    """
    rs = ResampleFeature(nb_points=points_count)
    trans_dataset = []
    for data in dataset:
        data = np.asarray([np.asarray(p) for p in data])
        trans_data = rs.extract(data)
        trans_dataset.append(trans_data)
    return trans_dataset


def get_line(point1, point2):
    """
    获取两点连成的直线的斜率和截距
    :param point1:
    :param point2:
    :return:
    """
    if point1[0] == point2[0]:
        return False, False
    k = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - k * point1[0]
    return k, b
