import math
import numpy as np
import matplotlib.pyplot as plt
import random


def loadData():
    dataMat = []
    dataLabel = []
    with open('testSet.txt') as f:
        while True:
            data = f.readline()
            if not data:
                break
            data = data.strip().split()
            dataMat.append([1, float(data[0]), float(data[1])])
            dataLabel.append(int(data[-1]))
    return dataMat, dataLabel


def sigmod(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataSet, dataLabel):
    dataMatSet = np.matrix(dataSet)
    dataMatLabel = np.matrix(dataLabel).transpose()
    m, n = dataMatSet.shape
    w = np.ones((n, 1))
    alpha = 0.001
    maxCycle = 500
    for i in range(maxCycle):
        h = sigmod(dataMatSet * w)
        error = dataMatLabel - h
        w = w + alpha * dataMatSet.transpose() * error
    return w


def stocGradAscent(dataSet, dataLabel, numIter=150):
    m = len(dataSet)
    n = len(dataSet[0])
    weight = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(sum(dataSet[randIndex] * weight))
            error = dataLabel[randIndex] - h
            weight = weight + [alpha * error * i for i in dataSet[randIndex]]
            del (dataIndex[randIndex])
    return weight


def plotBestFilt(weights):
    dataMat, dataLabel = loadData()
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(dataMat)):
        if dataLabel[i] == 0:
            xcord0.append(dataMat[i][1])
            ycord0.append(dataMat[i][2])
        elif dataLabel[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.ravel((-weights[0] - weights[1] * x) / weights[2])
    print(x)
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVec(inX, weight):
    prob = sigmod(sum(inX * weight))
    if prob > 0.5:
        return 1
    return 0


def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        trainSet.append(list(map(float, line[:-1])))
        trainLabel.append(float(line[-1]))
    trainWeight = stocGradAscent(trainSet, trainLabel, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        line = line.strip().split('\t')
        lineArr = list(map(float, line[:-1]))
        if classifyVec(lineArr, trainWeight) != int(line[-1]):
            errorCount += 1
    print("the error rate is ", errorCount / numTestVec)


if __name__ == "__main__":
    dataSet, dataLabel = loadData()
    w = stocGradAscent(dataSet, dataLabel)
    print(w)
    plotBestFilt(w)
    colicTest()
