from numpy import *
from boost import *

def loadSimpData():
	"""
	测试数据
	Author: ljq
	Date: 2017.10.16
	"""
	datMat = matrix([[1., 2.1],
		[2. , 1.1],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ],])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return datMat,classLabels

#基于单层决策树的AdaBoost的训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	aggClassEst = mat(zeros((m,1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels,D)
		print('error: ',error)
		print('D:',D.T)
		print('classLabels: ',classLabels)
		alpha = float(0.5*log((1-error)/max(error,1e-16)))
		print('alpha: ',alpha)
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		print('classEst: ', classEst.T)
		#print('classLabels ',mat(classLabels))

		expon = multiply(-1*alpha*mat(classLabels).T,classEst)#行向量*列向量

		D = multiply(D, exp(expon))
		D = D/D.sum()
		#计算累加错误率
		aggClassEst += alpha*classEst
		print("aggClassEst: ", aggClassEst.T)
		aggErrors = multiply(sign(aggClassEst) !=\
					mat(classLabels).T, ones((m,1)))


		errorRate = aggErrors.sum()/m
		print("total error: ", errorRate,"\n")
		if errorRate == 0.0:
			break
	return weakClassArr, aggClassEst

#基于多个弱分类器进行分类
def adaClassify(datToClass, classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArray)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
								classifierArr[i]['thresh'],\
								classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		print(aggClassEst)
	return sign(aggClassEst)


#加载数据
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat


#ROC 曲线绘制
def plotROC(predStrengths, classLabels):
	import matplotlib.pyplot as plt 
	cur = (1.0,1.0)
	ySum = 0.0
	numPosClas = sum(array(classLabels) == 1.0)
	yStep = 1/float(numPosClas)
	xStep = 1/float(len(classLabels)-numPosClas)
	sortedIndicies = predStrengths.argsort()
	fig   = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep;
		else:
			delX = xStep; delY = 0;
			ySum += cur[1]
		ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
		cur = (cur[0]-delX, cur[1]-delY)

	ax.plot([0,1], [0,1], 'b--')
	plt.xlabel('False Positive Rate');
	plt.ylabel('True  Positive Rate');
	plt.title('ROC curve for AdaBoost Horse  Colic Detection System');
	ax.axis([0,1 ,0,1])
	plt.show()
	print("the Area Under the Curve is: ",ySum*xStep);



if __name__ == '__main__':
	#datMat, classLabels = loadSimpData()
	#classifierArray = adaBoostTrainDS(datMat,classLabels,9)
	#print(classifierArray)
	'''
	aggClassEst = adaClassify([0,0], classifierArray)
	aggClassEst = adaClassify([[5,5], [0,0]],classifierArray)
	print(aggClassEst)
	'''
	datArr, labelArr = loadDataSet('horseColicTraining2.txt')
	classifirerArray,aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
	plotROC(aggClassEst.T, labelArr )
	#testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
	#prediction10 = adaClassify(testArr, classifierArray)
	#errArr = mat(ones((67,1)))
	#errSum = errArr[prediction10!=mat(testLabelArr).T].sum()
	#print(errSum,errSum/67)