import numpy as np
import operator
import matplotlib.pyplot as plt
import os


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 扩展输入向量
    sqDiffMat = diffMat ** 2
    sqDisstances = sqDiffMat.sum(axis=1)
    distances = sqDisstances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1])
    return sortedClassCount[0][0]


def file2matrix(Path):
    with open(Path) as f:
        dataMat = []
        dataLabel = []
        while True:
            data = f.readline()
            if not data:
                break
            data = data.strip('\n')
            data = data.split('\t')
            data = list(map(float, data))
            dataMat.append(data[0:-1])
            dataLabel.append(data[-1])
        return np.array(dataMat), np.array(dataLabel)


def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVal


def datingClassTest():
    rate = 0.1
    DataSet, Label = file2matrix("datingTestSet2.txt")
    m = DataSet.shape[0]
    numTest = int(m * rate)
    normDataSet, ranges, minVal = autoNorm(DataSet)
    errorCount = 0
    for i in range(numTest):
        label = classify0(normDataSet[i, :], normDataSet[numTest:, :], Label[numTest:], 5)
        print("predict label is %d, real label is %d" % (label, Label[i]))
        if label != Label[i]:
            errorCount += 1
    print("the total error rate is %f" % (errorCount / numTest))


def img2Vec(filename):
    returnVec = np.zeros((1, 32 * 32))
    with open(filename) as f:
        for i in range(32):
            data = f.readline()
            for j in range(32):
                returnVec[0, i * 32 + j] = data[j]
    return returnVec


def handwritingClassTest():
    files = os.listdir("trainingDigits")
    m = len(files)
    labels = []
    trainDataset = np.zeros((m, 32 * 32))
    for i in range(m):
        labels.append(int(files[i].split('_')[0]))
        trainDataset[i, :] = img2Vec("trainingDigits/" + files[i])
    testFiles = os.listdir("testDigits")
    mTest = len(testFiles)
    errorCount = 0
    for i in range(mTest):
        Label = int(testFiles[i].split('_')[0])
        testDataSet = img2Vec("testDigits/" + testFiles[i])
        predictLabel = classify0(testDataSet, trainDataset, labels, 3)
        print("the predict label is %s, the real label is %s" %
              (predictLabel, Label))
        if predictLabel != Label:
            errorCount += 1
    print("the error rate is %f" % (errorCount / mTest))


if __name__ == "__main__":
    # dataMat, dataLabel = file2matrix("datingTestSet2.txt")
    # fig = plt.figure(figsize=(6, 4))
    # # s表示数据点大小，c表示颜色序列
    # plt.scatter(dataMat[:, 0], dataMat[:, 1], s=15 * dataLabel, c=15 * dataLabel)
    # plt.show()
    # datingClassTest()
    handwritingClassTest()
