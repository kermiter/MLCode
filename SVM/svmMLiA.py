import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
import pdb

#加载数据集
def loadDataSet(fileName):
    dataMat = []; labelMat = [];
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#随机调整J值

def selectJrand(i,m):
    j=i
    while j==i:
        j = int(random.uniform(i,m))
    return j

#调整alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H;
    if aj < L:
        aj = L;
    return aj

#简化版smo算法.
#Input :
#dataMatIn 输入矩阵
#classLabels: 分类标签
#C:常数C
#toler:容忍率
#maxIter:最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose();
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    Iter = 0
    while ( Iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*\
                        (dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i]*Ei > toler) and\
                (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]+C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L == H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T-\
                      dataMatrix[j,:]*dataMatrix[i,:].T-\
                      dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta >= 0"); continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold) < 0.00001): print("j not move enough"); continue
                alphas[i] += labelMat[j]*labelMat[j]*\
                             (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[i,:].T -\
                     labelMat[j]*(alphas[j]-alphaJold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T
                bs = b1.copy()
                
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T - \
                     labelMat[j]*(alphas[j] - alphaJold)*\
                     dataMatrix[j, :]*dataMatrix[j,:].T
                bs = b1.copy()
                if (0 < alphas[i]) and (C > alphas[i]):  b = b1
                elif (0 < alphas[j])and (C > alphas[j]): b = b2
                else:
                    #pdb.set_trace()
                    b = (bs + b2)/2.0 
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %\
                      (Iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0): Iter += 1
        else: Iter = 0

        print("iteration number: %d" % Iter)
    return b,alphas
#################完整版Platt SMO的支撑函数################
def kernelTrans(X, A, kTup):
    """
    核转换函数
    k(x,y) = exp(-||z-y||2/2@2)
    """
    m,n = np.shape(X) 
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:]-A
            #print("m:%d,n:%d"%shape(deltaRow))
            #pdb.set_trace()
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem --\
            That kernel is not recognized')
    return K


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn #输入矩阵
        self.labelMat = classLabels #分类标签
        self.C = C #惩罚因子
        self.tol = toler #容忍度
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for  i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:],kTup)

#计算误差
"""
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*\
            oS.X*oS.X[k,:].T)+oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
"""
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek  =  fXk - float(oS.labelMat[k])
    return Ek
#随机选择J    
def selectJ(i, oS, Ei):
    """
    作者：LJQ
    """
    maxK = -1; maxDeltaE = 0; Ej = 0;
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == 1: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                Ej = Ek
                maxDeltaE = deltaE
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)    
    return j, Ej   
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]


#第二个alpha 选择中启发式方法        
def innerL(i, oS):
    #if 1:
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H"); 
            return 0
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i,:] * oS.X[i, :].T -\
        #     oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]

        ##-eta = K11 + K22 - 2K12 = ||@1-@2||2  >=0,则 eta <= 0其中@1代表核函数     
        if eta >= 0: 
            print("eta >= 0");
            return 0 
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j]  = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001 ):
            print("j 变化太小"); 
            return 0;
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
                        (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        #b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold) * \
        #     oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j]*\
        #     (oS.alphas[j] - alphaJold)*oS.X[i, :]*oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[j,j]
        #b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)* \ 
        #     oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*\
        #     (oS.alphas[j] - alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
            oS.b = b2
        else:
         oS.b = (b1 + b2)/2.0
    
        return 1
    else:
        return 0

#完整版Platt SMO的外循环代码
def  smoP(dataMatIn, classLabels, C, toler, maxIter, kTup =('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)
    Iter = 0
    entireSet = True; alphaPairsChanged = 0;
    while((Iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet))):
        alphaPairsChanged = 0
        if entireSet :
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter:%d i:%d, pairs changed %d' %\
                    (Iter, i, alphaPairsChanged))
            Iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i: %d , pairs changed %d"%\
                 (Iter, i, alphaPairsChanged))
            Iter += 1
        if entireSet: entireSet = False
        elif(alphaPairsChanged == 0): 
            entireSet = True
            print("iteration number: %d" %Iter)
    return oS.b, oS.alphas



def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    W = np.zeros((n,1))

    for i in range(m):
        W += np.multiply(labelMat[i]*alphas[i],X[i,:].T)
    return W

def showClassifer(dataMat, classLabels, w, b,alphas):
    """
    分类结果可视化
    Parameters:
        dataMat - 数据矩阵
        w - 直线法向量
        b - 直线解决
    Returns:
        无
    """
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    #y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    #plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()            
            
def plotClass(alphas, dataArr, classLabels,b):
    w = calcWs(alphas, dataArr, classLabels);
    showClassifer(dataArr, classLabels, w, b,alphas);
    return
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf',k1))
    plotClass(alphas, dataArr, labelArr,b)
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0] #获取支撑向量索引
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors"%np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!= np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f"%(float(errorCount/m)))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorConut = 0
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!= sign(labelArr[i]): 
            errorCount += 1
    print("the best error rate is: %f" % (float(errorCount)/m))














if __name__ =='__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    #b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b=0
    #alphas = np.zeros((len(dataArr),1))
    #b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    testRbf()
   #w = calcWs(alphas, dataArr, labelArr)
   # showClassifer(dataArr,labelArr,w,b)