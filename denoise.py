from deletePoints import *
from sklearn.cluster import DBSCAN
EPS = 0.1
MIN_SAMPLES = 2

# 9 remove

dbscan_parms_dict = {
    14 : (0.3, 2),
    3 : (0.1,2),
    9 : ()
}


file = "E:\\a_school\\books\\大三下\\挖掘\\challenge\\gps\\training_data\\9.txt"
dataset = readTrajectoryDataset(file)


def dbscan_preprocess(point_trajs):
    points = []
    for i in range(len(point_trajs)):
        for j in range(len(point_trajs[i])):
            points.append(point_trajs[i][j])


    # return points, p_idx_dict, p_idx
    return points

def main():
    data = format_dataset(dataset)
    result = delete_point_all(data)
    points = dbscan_preprocess(result)
    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    model.fit(points)
    labels = model.labels_
    remove_points = []
    for i in range(len(labels)):
        if labels[i] == -1:
            remove_points.append(points[i])

    result_save = []
    for traj in result:
        tmp = []
        for p in traj:
            if p not in remove_points:
                tmp.append(p)
        if len(tmp) >= 2:
            result_save.append(tmp)

    return result_save, result


if __name__ == "__main__":
    result, origin = main()
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    for traj in origin:
        x = []
        y = []
        for point in traj:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, c='b')
        plt.scatter(x, y, c='b')
    for traj in result:
        x = []
        y = []
        for point in traj:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, c= 'r')
        plt.scatter(x, y, c='r')


    plt.show()
