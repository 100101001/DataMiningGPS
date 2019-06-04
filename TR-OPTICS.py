import matplotlib.pyplot as plt
import numpy as np
import math
from pythonSample import *
import matplotlib.pyplot as plt


file = "E:\\a_school\\books\\大三下\\挖掘\\challenge\\gps\\training_data\\5.txt"
dataset = readTrajectoryDataset(file)
# OPTICS参数设定
epsilon = 0.1
minPts = 5
CORNER_ANGLE = math.pi / 4


class Line:
    """
    自定义线段类
    属性：
    两个端点的x, y值；线段长度len；线段所在直线的斜率k、截距b、角度angle以及直线是否垂直vertical（boolean）
    """
    def __init__(self,p1,p2):
        self.x1=p1[0]
        self.y1=p1[1]
        self.x2 = p2[0]
        self.y2 = p2[1]
        self.len=math.sqrt(math.pow((self.x1-self.x2),2)+math.pow((self.y1-self.y2),2))


        if self.x2-self.x1==0:
            self.vertical=True
            self.k=0
            self.b=0
            self.angle=math.pi/2
        else:
            self.vertical = False
            self.k=(self.y2-self.y1)/(self.x2-self.x1)
            self.b=-self.k*self.x1+self.y1
            self.angle=math.atan(self.k)

    def __str__(self):
        return "[({},{})-->({},{})]".format(self.x1, self.y1, self.x2, self.y2)





    def similarity(self,line):
        """
        定义两线相似性
        :param line:
        :return: 平行距离+垂直距离+角度距离
        """
        # 两线段夹角
        angle = GetCrossAngle(self, line)
        # horizontal distance
        dh=(max(self.len,line.len)-min(self.len,line.len)*math.cos(angle))/2
        x_project_1=0
        x_project_2 = 0
        y_project_1=0
        y_project_2=0

        # 选择两条线中更长的作为直线，求得线段在该直线上的两个映射点坐标
        if self.len >= line.len:
            k = self.k
            if self.vertical!=True and k!=0:
                x_project_1 = (line.x1+k*line.y1-k*self.y1+k*k*self.x1)/(k*k+1)
                y_project_1 = (k*k*line.x1-line.y1*k+k*self.y1-k*k*self.x1)/(k*k*k+k)+line.y1
                x_project_2 = (line.x2 + k * line.y2 - k * self.y1 + k * k * self.x1) / (k * k + 1)
                y_project_2 = (k * k * line.x2 - line.y2 * k + k * self.y1 - k * k * self.x1) / (k * k * k + k) + line.y2
            if self.vertical!=True and k==0:
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
            k=line.k
            if line.vertical != True and k != 0:
                x_project_1 = (self.x1 + k * self.y1 - k * line.y1 + k * k * line.x1) / (k * k + 1)
                y_project_1 = (k * k * self.x1 - self.y1 * k + k * line.y1 - k * k * line.x1) / (k * k * k + k) + self.y1
                x_project_2 = (self.x2 + k * self.y2 - k * line.y1 + k * k * line.x1) / (k * k + 1)
                y_project_2 = (k * k * self.x2 - self.y2 * k + k * line.y1 - k * k * line.x1) / (k * k * k + k) + self.y2
            if line.vertical!=True and k==0:
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
            dh = (math.sqrt(math.pow(self.x1-x_project_1,2)+math.pow(self.y1-y_project_1,2))+math.sqrt(math.pow(self.x2-x_project_2,2)+math.pow(self.y2-y_project_2,2)))/2
        else:
            dh = (math.sqrt(math.pow(line.x1-x_project_1,2)+math.pow(line.y1-y_project_1,2))+math.sqrt(math.pow(line.x2-x_project_2,2)+math.pow(line.y2-y_project_2,2)))/2
        #print("horizontal distance: " + str(dh))

        # vertical distance
        dv=0
        if(self.len<=line.len):
            dv= (GetPointToLineDistance(line.x1,line.y1,line.x2,line.y2,self.x1,self.y1)+GetPointToLineDistance(line.x1,line.y1,line.x2,line.y2,self.x2,self.y2))/2
        else:
            dv = (GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x1,
                                        line.y1) + GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x2,
                                        line.y2)) / 2
        #print("vertical distance: "+str(dv))

        # angle range
        dtheta=0
        if angle<=180 and angle >90:
            dtheta=max(self.len,line.len)
        else:
            dtheta = min(self.len, line.len)*math.sin(angle)
        #print("angle range: "+str(dtheta))

        # 两线段相似度，三种距离之和
        distance=dh+dv+dtheta
        #print("distance:"+str(distance))
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
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # 注意转成浮点数运算
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
    return np.sqrt((array_vec - array_temp).dot(array_vec - array_temp))


def points2line(dataset):
    """
    把轨迹文件的点转换成线段
    :param dataset: 一个轨迹文件，字典（点）数组
    :return: 文件中包含的所有轨迹分的线段数组
    """
    lines=[]
    for trajectory in dataset:
        for i in range(0,len(trajectory)-1):
            p1 = [trajectory[i]['x'],trajectory[i]['y']]
            p2 = [trajectory[i+1]['x'], trajectory[i+1]['y']]
            lines.append(Line(p1,p2))

    for trajectory in dataset:
        xs=[]
        ys=[]
        for i in range(0, len(trajectory) - 1):
            xs.append(trajectory[i]['x'])
            ys.append(trajectory[i]['y'])
        plt.plot(xs,ys)
    return lines


def distanceMatrix(lines):
    """
    计算线段两两的相似度
    :param lines: 一个文件的所有线段(轨迹分的)
    :return: dm[i][j] 代表 线段 i 与 线段 j+i+1 的相似度。i的取值范围是 0到线段数量-1，j的范围参考左上三角形。
    """
    dm = []
    for i in range(0, len(lines)):
        dm_i=[]
        for j in range(i + 1, len(lines)):
            dm_i.append(lines[i].similarity(lines[j]))
        dm.append(dm_i)
    return dm


def searchNeighbors(line, x0, y0, epsilon):
    """
    判断点是否在某条线段的矩形领域中
    :param line: Line 对象
    :param x0:
    :param y0:
    :return: Boolean
    """
    # 当线段不是水平、垂直的
    if line.vertical == False and line.k != 0:
        b3 = line.b + epsilon/abs(math.cos(line.angle))
        b4 = line.b - epsilon/abs(math.cos(line.angle))
        b1 = 0
        b2 = 0
        if line.y1 < line.y2:
            b1 = (1/line.k)*line.x1+line.y1-epsilon/math.sin(line.angle)
            b2 = (1/line.k)*line.x2+line.y2+epsilon/math.sin(line.angle)
        else:
            b1 = (1 / line.k) * line.x2 + line.y2 - epsilon / math.sin(line.angle)
            b2 = (1 / line.k) * line.x1 + line.y1 + epsilon / math.sin(line.angle)

        if  line.k*x0+b4 < y0 and line.k*x0+b3 > y0 and (-1/line.k)*x0+b1 < y0 and (-1/line.k)*x0+b2 > y0:
            return True
        return False
    # 线段垂直
    if line.vertical == True:
        y1 = 0
        y2 = 0
        if line.y1>line.y2:
            y1 = line.y2
            y2 = line.y1
        else:
            y1 = line.y1
            y2 = line.y2
        if x0 < (line.x1+epsilon) and x0 > (line.x1-epsilon) and y0 < (y2+epsilon) and y0 > (y1-epsilon):
            return True
        return False
    # 线段水平
    x1 = 0
    x2 = 0
    if line.x1 > line.x2:
        x1 = line.x2
        x2 = line.x1
    else:
        x1 = line.x1
        x2 = line.x2
    if y0 < (line.y1 + epsilon) and y0 > (line.y1 - epsilon) and x0 < (x2 + epsilon) and x0 > (x1 - epsilon):
        return True
    return False


def create_adj_matrix(lines, dm, epsilon):
    """
    创建邻接表并返回，基表的邻表是搜索范围内的线段
    :param lines: 线段数组
    :return: 创建好的邻接表
    """
    adjacent_matrix = []
    for i in range(len(lines)):
        adjacent_matrix_i=[]
        for j in range(len(lines)):
            if i != j:
                if searchNeighbors(lines[i],lines[j].x1,lines[j].y1,epsilon)==True or searchNeighbors(lines[i],lines[j].x2,lines[j].y2,epsilon)==True:
                    if i < j:
                        adjacent_matrix_i.append([dm[i][j - i - 1],j])
                    else:
                        adjacent_matrix_i.append([dm[j][i - j - 1], j])
        print("领域范围内的轨迹号")
        print(i)
        print(adjacent_matrix_i)
        adjacent_matrix.append(adjacent_matrix_i)
    return adjacent_matrix


def cal_core_and_reachable_dist(adj_matrix,dm):
    """
    计算核心轨迹的核心距离，以及轨迹间的可达距离，更新邻接表为可达距离
    :param adj_matrix: 邻接表
    :param dm: 距离矩阵
    :return: 核心距离，可达距离，邻接表
    """

    # core_distance[i] 代表使核心轨迹i成为核心对象的距离
    core_distance=[]
    for i in range(len(adj_matrix)):
        if len(adj_matrix[i]) > minPts:
            sorted_adj=sorted(adj_matrix[i])
            core_distance.append(sorted_adj[minPts-1][0])
        else:
            core_distance.append(-1)
    print("核心距离")
    print(core_distance)
    # reachable_distance[i][j] 代表轨迹i关于轨迹j的核心可达距离; 取核心距离与两者相似度的较大值
    reachable_distance=[]
    for i in range(len(adj_matrix)):
        reachable_distance_i=[]
        for j in range(len(adj_matrix)):
            if i!=j:
                if i < j:
                    reachable_distance_i.append(max(core_distance[i], dm[i][j - i - 1]))
                else:
                    reachable_distance_i.append(max(core_distance[i], dm[j][i - j - 1]))
            else:
                reachable_distance_i.append(max(core_distance[i],0))
        reachable_distance.append(reachable_distance_i)

    print("可达距离")
    print(reachable_distance)

    # 使用核心可达距离更新邻接表,是核心对象直接可达领域范围内对象的距离，还是领域范围内对象到核心对象的核心可达距离?
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if i!=j:
                adj_matrix[i][j][0]=reachable_distance[adj_matrix[i][j][1]][i]

    # adjacent_matrix=[]
    # for i in range(len(adj_matrix)):
    #     if len(adj_matrix[i])>minPts:
    #         adjacent_matrix.append(adj_matrix[i])

    return core_distance,reachable_distance,adj_matrix


def findElemInTupleList(list, value, index):
    """
    找到目标元组在数组中的位置
    :param list: 数组
    :param value: 目标值
    :param index: 元组的第几个元素
    :return: 目标位置，如果不存在返回 -1
    """
    i=0
    for tuple in list:
        if tuple[index] == value:
            return i
        i=i+1
    return -1


def TR_OPTICS(lines, adjacent_matrix, minPts):
    """
    TR-OPTICS核心算法
    :param lines: 包含所有线段的数组
    :param adjacent_matrix: 邻接表
    :param minPts: 限定邻域内的最少点数
    :return: 结果数组O
    """
    # 有序集合S，结果集O
    S = []
    O = []
    for i in range(len(adjacent_matrix)):
        if len(S) == 0:
            pts = adjacent_matrix[i]

            if len(pts) > minPts:
                O.append([0,i])
                for s in adjacent_matrix[i]:
                    if s[1] not in O:
                        if findElemInTupleList(S,s[1],1)==-1:
                            S.append(s)
                        else:
                            S[findElemInTupleList(S, s[1], 1)][0] = s[0]
                S.sort()
        else:
            i = i - 1
            s0 = S[0]
            del S[0]
            pts = adjacent_matrix[s0[1]]
            if len(pts) > minPts:
                O.append(s0)
                for s in adjacent_matrix[s0[1]]:
                    if s[1] not in O:
                        if findElemInTupleList(S, s[1], 1) == -1:
                            S.append(s)
                        else:
                            S[findElemInTupleList(S, s[1], 1)][0] = s[0]
                S.sort()
    return O


def points2line_2(dataset):
    """
    把轨迹文件的点转换成线段
    :param dataset: 一个轨迹文件，字典（点）数组
    :return: 文件中包含的所有轨迹分的线段数组
    """
    lines=[]
    for trajectory in dataset:
        line=[]
        for i in range(0,len(trajectory)-1):
            p1 = [trajectory[i]['x'],trajectory[i]['y']]
            p2 = [trajectory[i+1]['x'], trajectory[i+1]['y']]
            line.append(Line(p1,p2))
        lines.append(line)

    # for trajectory in dataset:
    #     xs=[]
    #     ys=[]
    #     for i in range(0, len(trajectory) - 1):
    #         xs.append(trajectory[i]['x'])
    #         ys.append(trajectory[i]['y'])
    #     plt.plot(xs,ys)
    return lines


def delete_point(line1, line2):
    # 两线都不垂直于x轴，斜率都存在
    theta = GetCrossAngle(line1, line2)
    print(theta * 180 / math.pi)
    if theta < CORNER_ANGLE:
        return [Line([line1.x1, line1.y1], [line2.x2, line2.y2])]  # 转角小于 pi / 4, 删中间点，构造新线段返回
    else:
        return [line1, line2]  # 转角大于 pi / 4, 不删点，直接返回两条原来的输入线段


def draw_traj(traj, color):
    x = []
    y = []
    for l in traj:
        x.append(l.x1)
        x.append(l.x2)
        y.append(l.y1)
        y.append(l.y2)

    plt.plot(x, y, c = color)


def test_delete_point():
    lines = points2line_2(dataset)
    new_lines = []

    for i in delete_point(lines[0][0], lines[0][1]):
        new_lines.append(i)
    i = 2
    while i < len(lines[0]):
        line1 = new_lines[-1]
        line2 = lines[0][i]
        lines_return = delete_point(line1, line2)
        if len(lines_return) == 2:
            new_lines.append(line2)
        if len(lines_return) == 1:
            new_lines[-1] = lines_return[0]
        i += 1

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    draw_traj(lines[0], 'b')
    draw_traj(new_lines, 'r')

    plt.show()


def main():
    lines = points2line(dataset)
    print(len(lines))
    dm = distanceMatrix(lines)
    am = create_adj_matrix(lines, dm, epsilon)
    core_dist, reachable_dist, adj_matrix = cal_core_and_reachable_dist(am,dm)
    O = TR_OPTICS(lines, adj_matrix, minPts)

    print(len(O))
    print([o[0] for o in O])
    result = []
    j = 0
    for o in O:
        result.append([o[1], j])
    print(result)
    x = []
    y = []
    for r in result:
        x.append(lines[r[0]].x1)
        x.append(lines[r[0]].x2)
        y.append(lines[r[0]].y1)
        y.append(lines[r[0]].y2)
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()
    plt.plot(range(len(O)), [o[0] for o in O])
    plt.scatter(range(len(O)), [o[0] for o in O], s=10, c='red')
    plt.show()


if __name__ == "__main__":
    # main()
    test_delete_point()
