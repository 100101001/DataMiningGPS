from utility import *
from test import *


# function that computes the road segment from a trajectory set
def computeAverageTrajectory(trajectorySet):
    # YOUR CODE SHOULD GO HERE
    # This demo returns the first trajectory in the set

    # 数据预处理
    trajectories = format_dataset(trajectorySet)

    # 取平均数的方法
    result = try_avg(dataset=trajectories, points_count=80)

    # 打印结果
    # print_result_graph(trajectories, result)
    # plt.show()

    # 结果转化为可输出的格式
    result = format_result(result)

    return result


# function reads all the datasets and returns each of them as part of an array
def readAllDatasets(inputDirectory):
    dataSets=[]
    import os
    for i in range(0, len(os.listdir(inputDirectory))):
        fileName = inputDirectory+"/"+str(i)+".txt"
        if(os.path.isfile(fileName)):
            dataSets.append(readTrajectoryDataset(fileName))
    return dataSets


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


# function for writing the result to a file
def writeSolution(generatedRoadSegments, outputFile):
    string=""
    for i in range(0, len(generatedRoadSegments)):
        segm = generatedRoadSegments[i]
        for j in range(0,len(segm)):
            string += "{:.7f}".format(segm[j]["x"])+" "+"{:.7f}".format(segm[j]["y"])+"\n"
        string += "\n"

    f = open(outputFile, "w+")
    f.write(string)
    f.close()


# MAIN
def main():
    inputDirectory = "./training_data"
    outputFile = "solution.txt"

    dataSets = readAllDatasets(inputDirectory)

    # 临时限制测试数
    dataSets = dataSets[:100]

    generatedRoadSegments = []
    for i in range(0, len(dataSets)):
        # print(str(i)+": ")
        generatedRoadSegments.append(computeAverageTrajectory(dataSets[i]))

    writeSolution(generatedRoadSegments, outputFile)


if __name__ == '__main__':
    main()
