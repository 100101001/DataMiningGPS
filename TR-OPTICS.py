import matplotlib.pyplot as plt
import numpy as np
import math


class Line:
    def __init__(self,p1,p2):
        self.x1=p1[0]
        self.y1=p1[1]
        self.x2 = p2[0]
        self.y2 = p2[1]
        self.len=math.sqrt(math.pow((self.x1-self.x2),2)+math.pow((self.y1-self.y2),2))

        if self.x2-self.x1==0:
            self.vertical=True
            self.k=0
        else:
            self.vertical = False
            self.k=(self.y2-self.y1)/(self.x2-self.x1)

    def __str__(self):
        return "[({},{})-->({},{})]".format(self.x1, self.y1,self.x2, self.y2)


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

        #
        if self.len>=line.len:
            # self 的斜率
            k=self.k
            if self.vertical!=True and k!=0:
                x_project_1=(line.x1+k*line.y1-k*self.y1+k*k*self.x1)/(k*k+1)
                y_project_1=(k*k*line.x1-line.y1*k+k*self.y1-k*k*self.x1)/(k*k*k+k)+line.y1
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

def GetPointToLineDistance(x1,y1,x2,y2,point_x,point_y):
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

def readTrajectoryDataset(fileName):
    """
    sample中的文件读取函数
    :param fileName:
    :return:
    """
    s = open(fileName, 'r').read();
    comp = s.split("\n")
    trajectory = [];
    trajectorySet = [];
    for i in range(0, len(comp)):
        comp[i] = comp[i].split(" ");
        if (len(comp[i]) == 2):
            # to double??
            point = {
                "x": float(comp[i][0]),
                "y": float(comp[i][1])
            }
            trajectory.append(point);
        else:
            trajectorySet.append(trajectory);
            trajectory = [];

    return trajectorySet;


#
# line1=Line([0,1],[1,1])
# line2=Line([1,2],[2,2])
# line1.similarity(line2)


file="E:\\a_school\\books\\大三下\\挖掘\\challenge\\gps\\training_data\\96.txt"

dataset=readTrajectoryDataset(file)

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
    return lines

lines=points2line(dataset)


def distanceMatrix(lines):
    """
    计算线段两两的相似度
    :param lines: 一个文件的所有线段(轨迹分的)
    :return: dm[i][j] 代表 线段 i 与 线段 j+i+1 的相似度。i的取值范围是 0到线段数量-1，j的范围参考左上三角形。
    """
    dm=[]
    for i in range(0, len(lines)):
        dm_i=[]
        for j in range(i + 1, len(lines)):
            dm_i.append(lines[i].similarity(lines[j]))
        dm.append(dm_i)
    return dm


dm=distanceMatrix(lines)

# OPTICS参数设定
epsilon=0.02
minPts=10

#邻接表 func
adjacent_matrix=[]
for i in range(len(lines)):
    adjacent_matrix_i=[]
    for j in range(len(lines)):
        if i!=j:
            if i<j:
                if dm[i][j - i - 1] <= epsilon:
                    adjacent_matrix_i.append([dm[i][j - i - 1],j])
            else:
                if dm[j][i-j-1]<=epsilon:
                    adjacent_matrix_i.append([dm[j][i-j-1],j])
    #print(adjacent_matrix_i)
    adjacent_matrix.append(adjacent_matrix_i)
print(adjacent_matrix[0][:5])
print(adjacent_matrix[100][:5])
print(len(lines))
# 有序集合S，结果集O
S=[]
O=[]

def findElemInTupleList(list,value,index):
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

#
for i in range(len(lines)):
    if len(S)==0:
        pts=adjacent_matrix[i]
        if len(pts)>minPts:
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


import matplotlib.pyplot as plt

plt.plot(range(len(O)),[o[0] for o in O])
plt.scatter(range(len(O)),[o[0] for o in O],s=10,c='red')
plt.show()

print(len(O))
print([o[0] for o in O])