import matplotlib.pyplot as plt
import numpy as np
import math
CORNER_ANGLE = math.pi / 5
from utility import *

# reads a set of trajectories from a file
def readTrajectoryDataset(fileName):
    """
    sample中的文件读取函数
    :param fileName:
    :return:
    """
    s = open(fileName, 'r').read()
    comp = s.split("\n")
    trajectory = []
    trajectorySet = []
    for i in range(0, len(comp)):
        comp[i] = comp[i].split(" ")
        if(len(comp[i]) == 2):
            # to double??
            point = {
                "x": float(comp[i][0]),
                "y": float(comp[i][1])
            }
            trajectory.append(point)
        else:
            trajectorySet.append(trajectory)
            trajectory = []

    return trajectorySet


# file = "E:\\a_school\\books\\大三下\\挖掘\\challenge\\gps\\training_data\\21.txt"
# dataset = readTrajectoryDataset(file)


def delete_point(line1, line2):
    # 两线都不垂直于x轴，斜率都存在
    theta = GetCrossAngle(line1, line2)
    if theta < CORNER_ANGLE:
        return [Line([line1.x1, line1.y1], [line2.x2, line2.y2])]  # 转角小于 pi / 4, 删中间点，构造新线段返回
    else:
        return [line1, line2]  # 转角大于 pi / 4, 不删点，直接返回两条原来的输入线段


def points2line_2(dataset):
    """
    把轨迹文件的点转换成
    :param dataset: 一个轨迹文件，字典（点）数组
    :return: 文件中包含的所有轨迹各自的组成线段组
    """
    lines = []
    for trajectory in dataset:
        line = []
        for i in range(0,len(trajectory)-1):
            p1 = [trajectory[i][0],trajectory[i][1]]
            p2 = [trajectory[i+1][0], trajectory[i+1][1]]
            line.append(Line(p1,p2))
        lines.append(line)

    return lines


class Line:
    """
    自定义线段类
    属性：
    两个端点的x, y值；线段长度len；线段所在直线的斜率k、截距b、角度angle以及直线是否垂直vertical（boolean）
    """

    def __init__(self, p1, p2):
        self.x1 = p1[0]
        self.y1 = p1[1]
        self.x2 = p2[0]
        self.y2 = p2[1]
        self.len = math.sqrt(math.pow((self.x1 - self.x2), 2) + math.pow((self.y1 - self.y2), 2))

        if self.x2 - self.x1 == 0:
            self.vertical = True
            self.k = 0
            self.b = 0
            self.angle = math.pi / 2
        else:
            self.vertical = False
            self.k = (self.y2 - self.y1) / (self.x2 - self.x1)
            self.b = -self.k * self.x1 + self.y1
            self.angle = math.atan(self.k)

    def __str__(self):
        return "[({},{})-->({},{})]".format(self.x1, self.y1, self.x2, self.y2)

    def similarity(self, line):
        """
        定义两线相似性
        :param line:
        :return: 平行距离+垂直距离+角度距离
        """
        # 两线段夹角
        angle = GetCrossAngle(self, line)
        # horizontal distance
        dh = (max(self.len, line.len) - min(self.len, line.len) * math.cos(angle)) / 2
        x_project_1 = 0
        x_project_2 = 0
        y_project_1 = 0
        y_project_2 = 0

        # 选择两条线中更长的作为直线，求得线段在该直线上的两个映射点坐标
        if self.len >= line.len:
            k = self.k
            if self.vertical != True and k != 0:
                x_project_1 = (line.x1 + k * line.y1 - k * self.y1 + k * k * self.x1) / (k * k + 1)
                y_project_1 = (k * k * line.x1 - line.y1 * k + k * self.y1 - k * k * self.x1) / (
                            k * k * k + k) + line.y1
                x_project_2 = (line.x2 + k * line.y2 - k * self.y1 + k * k * self.x1) / (k * k + 1)
                y_project_2 = (k * k * line.x2 - line.y2 * k + k * self.y1 - k * k * self.x1) / (
                            k * k * k + k) + line.y2
            if self.vertical != True and k == 0:
                x_project_1 = line.x1
                x_project_2 = line.x2
                y_project_1 = self.y1
                y_project_2 = self.y2
            if self.vertical == True:
                x_project_1 = self.x1
                x_project_2 = self.x2
                y_project_1 = line.y1
                y_project_2 = line.y2
        else:
            k = line.k
            if line.vertical != True and k != 0:
                x_project_1 = (self.x1 + k * self.y1 - k * line.y1 + k * k * line.x1) / (k * k + 1)
                y_project_1 = (k * k * self.x1 - self.y1 * k + k * line.y1 - k * k * line.x1) / (
                            k * k * k + k) + self.y1
                x_project_2 = (self.x2 + k * self.y2 - k * line.y1 + k * k * line.x1) / (k * k + 1)
                y_project_2 = (k * k * self.x2 - self.y2 * k + k * line.y1 - k * k * line.x1) / (
                            k * k * k + k) + self.y2
            if line.vertical != True and k == 0:
                x_project_1 = self.x1
                x_project_2 = self.x2
                y_project_1 = line.y1
                y_project_2 = line.y2
            if line.vertical == True:
                x_project_1 = line.x1
                x_project_2 = line.x2
                y_project_1 = self.y1
                y_project_2 = self.y2

        # 计算两种情况下的水平距离
        if self.len >= line.len:
            dh = (math.sqrt(math.pow(self.x1 - x_project_1, 2) + math.pow(self.y1 - y_project_1, 2)) + math.sqrt(
                math.pow(self.x2 - x_project_2, 2) + math.pow(self.y2 - y_project_2, 2))) / 2
        else:
            dh = (math.sqrt(math.pow(line.x1 - x_project_1, 2) + math.pow(line.y1 - y_project_1, 2)) + math.sqrt(
                math.pow(line.x2 - x_project_2, 2) + math.pow(line.y2 - y_project_2, 2))) / 2
        # print("horizontal distance: " + str(dh))

        # vertical distance
        dv = 0
        if (self.len <= line.len):
            dv = (GetPointToLineDistance(line.x1, line.y1, line.x2, line.y2, self.x1, self.y1) + GetPointToLineDistance(
                line.x1, line.y1, line.x2, line.y2, self.x2, self.y2)) / 2
        else:
            dv = (GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x1,
                                         line.y1) + GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x2,
                                                                           line.y2)) / 2
        # print("vertical distance: "+str(dv))

        # angle range
        dtheta = 0
        if angle <= 180 and angle > 90:
            dtheta = max(self.len, line.len)
        else:
            dtheta = min(self.len, line.len) * math.sin(angle)
        # print("angle range: "+str(dtheta))

        # 两线段相似度，三种距离之和
        distance = dh + dv + dtheta
        # print("distance:"+str(distance))
        return distance


def GetCrossAngle(l1, l2):
    """
    求两线段间夹角
    :param l1:
    :param l2:
    :return: 夹角弧度值
    """
    arr_0 = np.array([(l1.x2 - l1.x1), (l1.y2 - l1.y1)])
    arr_1 = np.array([(l2.x2 - l2.x1), (l2.y2 - l2.y1)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))  # 注意转成浮点数运算
    return np.arccos(cos_value)


def GetPointToLineDistance(x1, x2, y1, y2, point_x, point_y):
    """
    一个点到线段的距离
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param point_x:
    :param point_y:
    :return:
    """
    array_line  = np.array([x2-x1, y2-y1])
    array_vec = np.array([x2-point_x, y2-point_y])
    # 用向量计算点到直线距离
    array_temp = (float(array_vec.dot(array_line)) / array_line.dot(array_line))   # 注意转成浮点数运算
    array_temp = array_line.dot(array_temp)
    return  np.sqrt((array_vec - array_temp).dot(array_vec - array_temp))


def draw_traj(traj, color):
    x = []
    y = []
    for l in traj:
        x.append(l.x1)
        x.append(l.x2)
        y.append(l.y1)
        y.append(l.y2)

    plt.plot(x, y, c = color)


def traj_line2point(traj):
    """
    将一条用线段标识的轨迹变成由点标识的轨迹
    :param traj: 线段数组
    :return:
    """
    points = []
    points.append([traj[0].x1, traj[0].y1])
    for t in traj:
        points.append([t.x2, t.y2])
    return points


def delete_point_all(dataset):
   """
   对一个文件的所有轨迹提取核心点
   :param dataset: [[[x,y].......],.... ]
   :return:
   """
   lines = points2line_2(dataset)
   new_lines_all = []
   for i in range(len(lines)):
       new_lines = []
       # 首两个线段的处理
       if len(lines[i]) < 2:
           new_lines = lines[i]
       else:
           for k in delete_point(lines[i][0], lines[i][1]):
               new_lines.append(k)

           # 其余线段的处理
           j = 2
           while j < len(lines[i]):
               line1 = new_lines[-1]
               line2 = lines[i][j]
               lines_return = delete_point(line1, line2)
               if len(lines_return) == 2:
                   new_lines.append(line2)
               if len(lines_return) == 1:
                   new_lines[-1] = lines_return[0]
               j += 1

       # 线段数组转点数组
       new_points = traj_line2point(new_lines)
       new_lines_all.append(new_points)

   return new_lines_all


if __name__ == "__main__":
    data = format_dataset(dataset)
    result = delete_point_all(data)
    print(len(data))
    print(len(result))
    for t, d in zip(result,data):
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        x = []
        y = []
        a = []
        b = []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        for point in d:
            a.append(point[0])
            b.append(point[1])
        plt.plot(x, y)
        plt.plot(a, b)

        plt.show()




