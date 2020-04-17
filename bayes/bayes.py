import numpy as np
import re
import random
import feedparser


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s not in vocabList" % word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word %s not in vocablist" % word)
    return returnVec


def trainNB0(trainMatrix, trainCatrgory):
    numDocument = len(trainMatrix)
    numWord = len(trainMatrix[0])
    pAbusive = sum(trainCatrgory) / len(trainCatrgory)
    p1Num = np.ones(numWord)
    p0Num = np.ones(numWord)
    p1Denom = p0Denom = 2
    for i in range(numDocument):
        if trainCatrgory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive


def classifyNB(Vec2Classify, p1Vec, p0Vec, pClass1):
    p1 = sum(Vec2Classify * p1Vec) + pClass1
    p0 = sum(Vec2Classify * p0Vec) + (1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPost, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPost)
    trainMat = []
    for i in listOPost:
        trainMat.append(setOfWords2Vec(myVocabList, i))
    p0Vec, p1Vec, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print("this Doc classify as ", classifyNB(thisDoc, p1Vec, p0Vec, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print("this Doc classify as ", classifyNB(thisDoc, p1Vec, p0Vec, pAb))


def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


def spamText():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    traingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(traingSet)))
        testSet.append(traingSet[randIndex])
        del (traingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in traingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pAb = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVec, p1V, p0V, pAb) != classList[docIndex]:
            errorCount += 1
    print("the error rate is ", errorCount / len(testSet))


def calMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=lambda item: item[1], reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    doclist = []
    classlist = []
    fulltext = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        doclist.append(wordList)
        fulltext.extend(wordList)
        classlist.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        doclist.append(wordList)
        fulltext.extend(wordList)
        classlist.append(0)
    vocabList = createVocabList(doclist)
    top30Words = calMostFreq(vocabList, fulltext)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(5):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, doclist[docIndex]))
        trainClasses.append(classlist[docIndex])
    p0V, p1V, pAb = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2Vec(vocabList, doclist[docIndex])
        if classifyNB(wordVec, p1V, p0V, pAb) != classlist[docIndex]:
            errorCount += 1
    print("error rate is ", errorCount / len(testSet))
    return vocabList, p0V, p1V


if __name__ == "__main__":
    # testingNB()
    # mySent = 'This book is the best book on python or M.L. I have ever laid eyes upon.'
    # print(mySent.split())
    # regEx = re.compile(r'\W+')
    # print(re.split(r'\W+', mySent))
    # spamText()
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    # print(ny['entries'])
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda item: item[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF")
    for i in sortedSF:
        print(i[0])
    sortedNY = sorted(topNY, key=lambda item: item[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY")
    for i in sortedNY:
        print(i[0])
