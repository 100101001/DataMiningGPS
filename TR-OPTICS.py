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

    # 定义两线相似性
    def similarity(self,line):
        # two lines' θ
        angle = GetCrossAngle(self, line)
        # horizontal distance
        dh=(max(self.len,line.len)-min(self.len,line.len)*math.cos(angle))/2
        print("horizontal distance: " + str(dh))
        # vertical distance
        dv=0
        if(self.len<=line.len):
            dv= (GetPointToLineDistance(line.x1,line.y1,line.x2,line.y2,self.x1,self.y1)+GetPointToLineDistance(line.x1,line.y1,line.x2,line.y2,self.x2,self.y2))/2
        else:
            dv = (GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x1,
                                        line.y1) + GetPointToLineDistance(self.x1, self.y1, self.x2, self.y2, line.x2,
                                        line.y2)) / 2
        print("vertical distance: "+str(dv))

        # angle range
        dtheta=0
        if angle<=180 and angle >90:
            dtheta=max(self.len,line.len)
        else:
            dtheta = min(self.len, line.len)*math.sin(angle)

        print("angle range: "+str(dtheta))
        distance=dh+dv+dtheta
        print("distance:"+str(distance))
        return distance

def GetCrossAngle(l1, l2):
    arr_0 = np.array([(l1.x2 - l1.x1), (l1.y2 - l1.y1)])
    arr_1 = np.array([(l2.x2 - l2.x1), (l2.y2 - l2.y1)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # 注意转成浮点数运算
    return np.arccos(cos_value)

def GetPointToLineDistance(x1,y1,x2,y2,point_x,point_y):
    array_line  = np.array([x2-x1, y2-y1])
    array_vec = np.array([x2-point_x, y2-point_y])
    # 用向量计算点到直线距离
    array_temp = (float(array_vec.dot(array_line)) / array_line.dot(array_line))   # 注意转成浮点数运算
    array_temp = array_line.dot(array_temp)
    return  np.sqrt((array_vec - array_temp).dot(array_vec - array_temp))

def readTrajectoryDataset(fileName):
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



line1=Line([0,1],[1,1])
line2=Line([0,0],[1,0])
line1.similarity(line2)


file="../training_data/5.txt"

def readTrajectoryDataset(fileName):
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

#angle = GetCrossAngle(line1, line2)
#print(angle)