from numpy import *

#获取数据
'''
[[ 10.235186  11.321997]
 [ 10.122339  11.810993]
 [  9.190236   8.904943]
 ..., 
 [  9.854922   9.201393]
 [  9.11458    9.134215]
 [ 10.334899   8.543604]]
 '''
def loadDataSet(fineName,delim ='\t'):
	fr = open(fineName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr    = [list(map(float,line)) for line in stringArr]
	return mat(datArr)  
#pca降维算法
def pca(dataMat, topNfeat = 9999999):
	meanVals = mean(dataMat, axis=0)#取均值,按列取均值
	#print("meanVals",meanVals)
	meanRemoved = dataMat - meanVals #去除均值
	covMat = cov(meanRemoved, rowvar = 0)#方差
	#print("covMat",covMat)
	eigVals, eigVects = linalg.eig(mat(covMat))#计算特征值，特征向量
	print(eigVals)
	print("eigVects",eigVects)
	eigValInd = argsort(eigVals)#默认按升序，返回对应的索引
	print(eigValInd)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	#print("eigValInd",eigValInd)
	redEigVects = eigVects[:,eigValInd]
	print(redEigVects)
	lowDDataMat = meanRemoved*redEigVects
	reconMat    = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat
#比较源数据与将维后的数据
def pcaPlot(dataMat,reconMat):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker ='^', s =90)
	ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker ='o',s=50, c ='red')
	plt.show()

if __name__=="__main__":
	dataMat = loadDataSet('testSet.txt')
	old = shape(dataMat)
	print(old)
	#print(dataMat)
	#lowDDataMat, reconMat = pca(dataMat,1)
	#print(shape(lowDDataMat))
	#print(lowDDataMat)
	#print(reconMat)
	#print(help(cov))
	#pcaPlot(dataMat,reconMat)
	#print(help(linalg.eig))
	lowDDataMat, reconMat = pca(dataMat,2)
	pcaPlot(dataMat,reconMat)

	#print(help(argsort))