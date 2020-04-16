import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def calShan(dataSet):
    m = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    shan = 0
    for i in labelCount.keys():
        prob = float(labelCount[i] / m)
        shan -= prob * math.log(prob, 2)
    return shan


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeat = len(dataSet[0]) - 1
    baseEntropy = calShan(dataSet)
    bestFeat = 0
    bestEntropy = -1
    for i in range(numFeat):
        FeatList = [example[i] for example in dataSet]
        uniqueFeat = set(FeatList)
        newEntropy = 0
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet) / len(dataSet))
            newEntropy += prob * calShan(subDataSet)
        if baseEntropy - newEntropy > bestEntropy:
            bestEntropy = baseEntropy - newEntropy
            bestFeat = i
    return bestFeat


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatName = labels[bestFeat]
    del (labels[bestFeat])
    featValue = [example[bestFeat] for example in dataSet]
    uniqueFeatValue = set(featValue)
    myTree = {bestFeatName: {}}
    for value in uniqueFeatValue:
        subLabel = labels[:]
        myTree[bestFeatName][value] = createTree(splitDataSet
                                                 (dataSet, bestFeat, value), subLabel)
    return myTree


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


decisionNode = dict(boxstyle='sawtooth', fc="0.8")
leafNode = dict(boxstyle='round4', fc="0.8")
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeaf(myTree):
    numLeaf = 0
    firstStr = list(myTree.keys())[0]
    secondTree = myTree[firstStr]
    for key in secondTree.keys():
        if type(secondTree[key]).__name__ == 'dict':
            numLeaf += getNumLeaf(secondTree[key])
        else:
            numLeaf += 1
    return numLeaf


def getTreeDepth(myTree):
    maxDeth = 0
    firstStr = list(myTree.keys())[0]
    secondTree = myTree[firstStr]
    for key in secondTree.keys():
        if type(secondTree[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondTree[key])
        else:
            thisDepth = 1
        if thisDepth > maxDeth:
            maxDeth = thisDepth
    return maxDeth

def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeaf = getNumLeaf(myTree)
    depth = getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt = (plotTree.xOff+(1.0+float(numLeaf))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor="white")
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeaf(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



if __name__ == "__main__":
    # dataSet, labels = createDataSet()
    # myTree = createTree(dataSet, labels)
    # print(myTree)
    # # createPlot()
    # print(getTreeDepth(myTree))
    # print(getNumLeaf(myTree))
    # createPlot(myTree)
    lenses=[]
    with open("lenses.txt") as f:
        while True:
            inst=f.readline()
            if not inst:
                break
            lenses.append(inst.strip().split('\t'))
            
    print(lenses)
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)
