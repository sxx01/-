import numpy as np


def loadSimpleData():
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def strumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArry = np.ones((dataMat.shape[0], 1))
    if threshIneq == 'lt':
        retArry[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArry[dataMat[:, dimen] > threshVal] = -1.0
    return retArry


def buildStrump(dataArr, classLabels, D):
    dataMat = np.matrix(dataArr)
    labelMat = np.matrix(classLabels)
    m, n = dataMat.shape
    numStepSize = 10.0
    bestStrump = {}
    bestClassEst = np.matrix(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMax = dataMat[:, i].max()
        rangeMin = dataMat[:, i].min()
        stepSize = (rangeMax - rangeMin) / numStepSize
        for j in range(-1, int(numStepSize + 1)):
            for Inequ in ['lt', 'gt']:
                threshVal = rangeMin + j * stepSize
                predictVal = strumpClassify(dataMat, i, threshVal, Inequ)
                errArr = np.matrix(np.ones((m, 1)))
                errArr[predictVal == labelMat.T] = 0
                weightError = D.T * errArr
                print("split: dim: %d, thresh: %.2f, thersh ineqal: %s, the weighted error: %.3f" %
                      (i, threshVal, Inequ, weightError))
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictVal.copy()
                    bestStrump['dim'] = i
                    bestStrump['thresh'] = threshVal
                    bestStrump['ineq'] = Inequ
    return bestStrump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    dataMat = np.matrix(dataArr)
    labelMat = np.matrix(classLabels)
    weakClassArr = []
    m = dataMat.shape[0]
    D = np.matrix(np.ones((m, 1)) / m)
    aggClassEst = np.matrix(np.zeros((m, 1)))
    for i in range(numIt):
        bestStrump, error, classEst = buildStrump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-6)))
        bestStrump['alpha'] = alpha
        weakClassArr.append(bestStrump)
        print("classEst is: ", classEst)
        expon = np.multiply(-1 * alpha * np.matrix(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst is: ", aggClassEst.T)
        aggError = np.multiply(np.sign(aggClassEst) != np.matrix(classLabels).T, np.ones((m, 1)))
        errorRate = aggError.sum() / m
        print("the total error is ", errorRate)
        if errorRate == 0:
            break
    return weakClassArr


def adaClassify(dataToClass, classifierArr):
    dataMat = np.matrix(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = strumpClassify(dataMat, classifierArr[i].get('dim'),
                                  classifierArr[i].get('thresh'), classifierArr[i].get('ineq'))
        aggClassEst += classifierArr[i].get('alpha') * classEst
    return np.sign(aggClassEst)


def loadDataSet(filename):
    dataMat = []
    labelClass = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(float, line[:-1])))
        labelClass.append(float(line[-1]))
    return dataMat, labelClass


if __name__ == "__main__":
    # dataMat, classLabel = loadSimpleData()
    # classifyArr = adaBoostTrainDS(dataMat, classLabel, 9)
    # print(classifyArr)
    # print(adaClassify([0, 0], classifyArr))
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(dataArr, labelArr, 9)
    testArr, testLabel = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = np.matrix(np.ones((67, 1)))
    print(errArr[prediction10 != np.matrix(testLabel).T].sum())
