from numpy import *
import feedparser

#构造词汇
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
                    'to', 'stop', 'him'],
                   ['quit', 'busying','worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 代表侮辱性文字, 0代表正常言论
    return postingList, classVec

#创建不重复词汇列表
def createVocabList (dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

#转化词汇为向量,词集模型:以单词是否出现作为一个特征
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print("word:%s is not in my Vocabulary"%word)

    return returnVec

#转化为词汇为向量,词袋模型:以单词出现的次数来作为一个特征
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec       
    
#trainMatrix:矢量化文档,trainCategory:文档类别矢量
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)#获取训练文档个数
    numWords = len(trainMatrix[0])#每个文档词汇数目相同？
    pAbusive = sum(trainCategory)/float(numTrainDocs)#计算侮辱性文档概率
    #p0Num = zeros(numWords); p1Num = zeros(numWords)#
    p0Num = ones(numWords); p1Num = ones(numWords)#防止出现0概率
    #p0Denom = 0.0; p1Denom = 0.0
    p0Denom = 2.0; p1Denom = 2.0;
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#遍历侮辱性文档
            p1Num += trainMatrix[i]#侮辱性文档中,每个词汇的计数
            p1Denom += sum(trainMatrix[i])#所有侮辱性文档中，所有词汇求和
        else:
            p0Num += trainMatrix[i]#非侮辱性文档中,每个词汇的计数
            p0Denom += sum(trainMatrix[i])#所有非侮辱性文档中,所有词汇计数            
    p1Vect = log(p1Num/p1Denom)#防止出现下溢
    p0Vect = log(p0Num/p0Denom)#防止出现下溢

    return p0Vect,p1Vect,pAbusive

def trainMatrix(dataSet,vocabList):
    "转化训练矩阵"
    trainMat = []
    for postinDoc in dataSet:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))
    return trainMat

#规则:p(w|c1)*p(c1)<p(w|c0)p(c0)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)

    if p1>p0:
        return 1
    else:
        return 0
#测试   
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as ', classifyNB(thisDoc, p0V, p1V, pAb))
#文本切割,正则表达式    
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    #随机构建测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    #剩余作为训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #验证测试集
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V,pSpam) !=\
        classList[docIndex]:
            errorCount +=1
            print('classification error', docList[docIndex])
            
    print('the error rate is:',float(errorCount)/len(testSet))
#RSS源分类器
#计算频率
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1),\
                        reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen  = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet = []
    #随机挑选测试集
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClass =[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(array(trainMat),array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) !=\
           classList[docIndex]:
            errorCount +=1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0v,p1v

    
    
if __name__ == '__main__':
    #listOPosts,listClasses = loadDataSet()
    #myVocabList = createVocabList(listOPosts)
    #print(myVocabList)
    #myWord2Vec = setOfWords2Vec(myVocabList,listOPosts[0])
    #print(myWord2Vec)
    #trainMat = trainMatrix(listOPosts,myVocabList)
    #print(trainMat)
    #p0v,p1v,pAb = trainNB0(trainMat,listClasses)    
    #print(p0v)
    #print(p1v)
    #print(pAb)
    '原始贝叶斯'
    #testingNB()
    '垃圾邮件分类'
    #spamTest()
    'RSS分类'
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, pSF,pNY = localWords(ny,sf)
    
