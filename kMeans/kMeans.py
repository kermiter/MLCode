from numpy import *
import matplotlib.pyplot as plt

'''
	testSet.txt:
		1.658985	4.285136
		-3.453687	3.424321
		4.838138	-1.151539
		-5.379713	-3.362104
		0.972564	2.924086
		-3.567919	1.531611
		0.450614	-3.302219
		-3.487105	-1.724432
		2.668759	1.594842
		-3.156485	3.191137
		3.165506	-3.999838
		-2.786837	-3.099354
'''
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float,curLine))
		#fltLine = float(cur(Line)
		dataMat.append(fltLine)
	return mat(dataMat)

#计算向量的欧式距离
def distEclud(vecA,vecB):
	return sqrt(sum(square(vecA - vecB)))

#随机取k个质点
def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ   = min(dataSet[:,j])#求取每一列的最小值:[[-5.379713],[-4.232586]],返回矩阵
		#print(type(minJ),minJ)
		rangeJ = float(max(dataSet[:,j]) - minJ)#求取最大与最小值差
		#print(rangeJ) 
		centroids[:,j] = minJ + rangeJ*random.rand(k,1)#rand:产生shape=(k,1)的随机矩阵，数值范围[0,1],k即为质点个数，在每一列上产生k个随机值，此处为k*1的矩阵。
		#print(help(random.rand))
		#print(centroids[:,j])
	#print(centroids)
	return centroids
#普通的K-Means方法
def kMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssMent = mat(zeros((m,2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				distJI = distMeans(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI ; minIndex = j
			if clusterAssMent[i,0] != minIndex:
				clusterChanged = True
			clusterAssMent[i,:] = minIndex, minDist**2
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssMent[:,0].A == cent)[0]]#根据质心为cent的元素的索引:获取每一个行向量
			centroids[cent,:] = mean(ptsInClust, axis=0)#计算每一个行向量的每一个维度上的均值:即通过按行进行计算每一个向量的均值
	cluster = []
	for i in range(k):
		cluster.append(dataSet[nonzero(clusterAssMent[:,0].A == i)[0]])

	return centroids, clusterAssMent 
#二分K-Means算法
def biKmeans(dataSet, k, distMeans = distEclud):
	m = shape(dataSet)[0]
	clusterAssMent = mat(zeros((m,2)))
	centroids0 = mean(dataSet, axis =0).tolist()[0]
	cenList = [centroids0]
	#构建初始堆
	for j in range(m):
		clusterAssMent[j,1] = distMeans(dataSet[j,:],mat(centroids0))

	while(len(cenList) < k):
		lowestSSE = inf
		for i in range(len(cenList)):
			ptsInCurrCluster =\
					dataSet[nonzero(clusterAssMent[:,0].A == i)[0],:]#获取每一个簇
			centoridMat, splitClusAss = \
					kMeans(ptsInCurrCluster, 2, distMeans)
			sseSplit = sum(splitClusAss[:,1])
			sseNotSplit = \
			   sum(clusterAssMent[nonzero(clusterAssMent[0,:].A != i)[0],1])#书中为!=,算法为计算分类后的总误差与分类前的总误差相比
			print("sseSplit, and sseNotSplit:",sseSplit, sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE:#保存最好的分类
				bestCentToSplit = i
				bestNewCents = centoridMat
				bestClustAss = splitClusAss.copy()	
				lowestSSE = sseSplit+ sseNotSplit
		#更新簇的分配结果		
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] =\
				len(cenList)
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] =\
				bestCentToSplit
		print("the bestCentToSplit is : ",bestCentToSplit)
		print("the len of bestClustAss is:",len(bestClustAss))
		cenList[bestCentToSplit] = bestNewCents[0,:].tolist()
		#cenList存储的是质心的均值
		cenList.append(bestNewCents[1,:].tolist())
		'''
		clusterAssMent更新过程:
			z  SSE              z   SSE
		[					[
			[0, 0.2],     		[0, 0.2],
			[0, 0.2],           [0, 0.2],
			[0, 0.1], ---->     [1, 0.1],
			[0, 0.3],           [1, 0.1],
			[0, 0.4]            [1, 0.1],
		]					]

		'''
		clusterAssMent[nonzero(clusterAssMent[:,0].A == \
							bestCentToSplit)[0],:] = bestClustAss
		#print("cenList:",matrix(cenList))
		#矩阵必须是二维的
		cenListTmp =[]
		for i in range(len(cenList)):
			cenListTmp.append(cenList[i][0])
		#print(cenList)

	return cenListTmp, clusterAssMent


def clusterPlot(dataMat,centroids,clusterAssMent):
	fig  = plt.figure(1)
	numClust = shape(centroids)[0]
	ax = plt.subplot(111)
	rect = [0.1, 0.1, 0.8, 0.8]
	scatterMarker = ['s','o','^','8','p','d','v','h','>','<']
	for i in range(numClust):
		ptsInCurrCluster = dataMat[nonzero(clusterAssMent[:,0].A == i)[0],:]
		markerStyle      = scatterMarker[i%len(scatterMarker)]
		ax.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
				   ptsInCurrCluster[:,1].flatten().A[0],\
				   marker = markerStyle, s=90)
	print("ptsInCurrCluster:",type(ptsInCurrCluster))
	print("centroids",type(centroids))
	#print("m:",shape(centroids)[2])
	
	#print(mat(centroids))
	centMat= mat(centroids)
	ax.scatter((centMat[:,0].flatten().A[0]),\
			   (centMat[:,1].flatten().A[0]),marker='+',s=300)
	
	plt.show()











if __name__ == "__main__":
	dataMat  = loadDataSet('testSet.txt')
	dataMat2 = loadDataSet('testSet2.txt')
	#print(shape(dataMat))
	#print(help(square))
	#print(help(nonzero))
	#print(help(mean))
	#centroids = randCent(dataMat,2)
	centroids, clusterAssMent = kMeans(dataMat, 2, distEclud,randCent);
	#print(help(matrix))
	#print(cluster)
	centList, myNewAssments  = biKmeans(dataMat2,3)
	#print(centList)
	#print(help(plt.scatter))
	#print(shape(centList))
	clusterPlot(dataMat2,centList,myNewAssments)