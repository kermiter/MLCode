from numpy import *

#加载数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#sigmoid 函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升函数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #numpy中将列表装换为矩阵
    labelMat = mat(classLabels).transpose()#将分类列表转化为行矢量,再转置为列矢量
    m,n = shape(dataMatrix)#获取行,列数
    alpha = 0.001
    maxCycles = 500

    weighs = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weighs)
        error = (labelMat-h)
        weighs = weighs + alpha*dataMatrix.transpose()*error#为什么乘以error?
    return weighs

#随机梯度上升

def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = h-classLabels[i]
        weights = weights + alpha*dataMatrix[i]*error
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(n):
            alpha = 4/(1.0+j+i) + 0.01#每次调整alpha
            randIndex = int(random.uniform(0,len(dataIndex)))#改进1:随机索引梯度上升
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[randIndex]-h
            weights = weights*1.0 + alpha*error*array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights

#########################预测病马######################
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest  = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(array(trainingSet),trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights)) !=\
           int(currLine[21]):
            errorCount +=1
    errorRate = float(errorCount)/numTestVec;
    print('the error rate of this test is:%f' % errorRate)

    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f'\
          %(numTests, errorSum/numTests))


        
        
        
        
#画出数据集和Logistic 回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0, 3.0, 0.1) #numpy 数组，60*1
    #print(type(x))
    #y = arange(-3.0, 3.0, 0.1)
    #print(x)
    y =array((-weights[0]-weights[1]*x)/weights[2]).transpose()
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
    
    
if __name__ == '__main__':
    mySet,myLabels = loadDataSet()
    myWeighs = gradAscent(mySet,myLabels)
    myWeighs = stocGradAscent1(mySet,myLabels,150)
    #print(myWeighs)
    #plotBestFit(myWeighs)
    multiTest()
