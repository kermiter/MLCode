from numpy import *
from time import sleep
import json
import urllib.request as request
import urllib.error

'''
	author: ljq
	Date: 2017/11/08
	function：convert text to list
	数据格式：
	abalone.txt:
	1	0.455	0.365	0.095	0.514	0.2245	0.101	0.15	15
	1	0.35	0.265	0.09	0.2255	0.0995	0.0485	0.07	7
	-1	0.53	0.42	0.135	0.677	0.2565	0.1415	0.21	9
	1	0.44	0.365	0.125	0.516	0.2155	0.114	0.155	10
	0	0.33	0.255	0.08	0.205	0.0895	0.0395	0.055	7
	abx =[
			[1	0.455	0.365	0.095	0.514	0.2245	0.101	0.15],
			[1	0.35	0.265	0.09	0.2255	0.0995	0.0485	0.07],
			[-1	0.53	0.42	0.135	0.677	0.2565	0.1415	0.21],
			[1	0.44	0.365	0.125	0.516	0.2155	0.114	0.155],
			[0	0.33	0.255	0.08	0.205	0.0895	0.0395	0.055],
		]
	abY = [15 7 9 10 7]	

	ex0.txt:
	1.000000	0.067732	3.176513
	1.000000	0.427810	3.816464
	1.000000	0.995731	4.550095
	1.000000	0.738336	4.256571
	1.000000	0.981083	4.560815
'''
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) -1#获取特征个数，如上所示numFeat = 8；
	#print("numFeat%d"%numFeat)

	dataMat = []; labelMat = [];
	fr = open(fileName);

	for line in fr.readlines():
	#for line in range(2):	
		lineArr = []
		curLine = line.strip().split('\t')#读取一行，以Tab:'\t'键分割
		for i in range(numFeat):#
			lineArr.append(float(curLine[i]))
		#print(lineArr);	
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	#print(labelMat)

	return dataMat,labelMat

'''
	标准求回归问题
'''
def standRegres(xArr, yArr):
	xMat = mat(xArr); yMat = mat(yArr).T 
	xTx  = xMat.T * xMat
	if linalg.det(xTx) == 0 :
		print("This matrix is singular, cannot do inverse")
		return
	ws = xTx.I * (xMat.T*yMat)

	return ws
def lwlr(testPoint, xArr, yArr, k = 1.0):
	xMat = mat(xArr); yMat = mat(yArr).T
	m = shape(xMat)[0]
	weights = mat(eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx = xMat.T*(weights * xMat)
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse");
		return
	ws = xTx.I*(xMat.T * (weights*yMat))
	return testPolint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr, yArr, k)
	return yHat
#平方误差和
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def showLine(file,func,k = 1.0):
	import matplotlib.pyplot as plt
	xArr, yArr = loadDataSet(file)
	print(xArr);
	#print(yArr)
	print(xArr[0:2],yArr[0:2])
	ws = func(xArr,yArr)
	xMat = mat(xArr)
	yMat = mat(yArr)
	yHat = xMat*ws

	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	
	xCopy = xMat.copy()
	#xCopy.sort(0)
	yHat = xCopy*ws
	ax.plot(xCopy[:,1], yHat)

	plt.show()

'''
	岭回归
'''
def ridgeRegres(xMat, yMat, lam = 0.2):
	xTx   = xMat.T*xMat
	denom = xTx + eye(shape(xMat)[1])*lam
	if linalg.det(denom) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = denom.I * (xMat.T*yMat)
	return ws

def ridgeTest(xArr, yArr):
	xMat  = mat(xArr); yMat = mat(yArr).T#yArr默认是行向量，yMat是列向量
	yMean = mean(yMat, 0)#什么意思
	yMat  = yMat - yMean
	xMeans = mean(xMat, 0)
	xVar  = var(xMat, 0)
	xMat = (xMat - xMeans)/xVar
	numTestPts = 30
	wMat = zeros((numTestPts,shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yMat,exp(i-10))
		wMat[i,:] = ws.T
	return wMat

#前向逐步线性回归	
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
	xMat  = mat(xArr); yMat = mat(yArr).T
	yMean = mean(yMat,0);
	yMat  = yMat-yMean
	#xMat  = regularize(xMat)#该函数什么作用？
	xMean = mean(xMat,0)
	xVar  = var(xMat,0)
	xMat  = (xMat-xMean)/xVar
	m,n   = shape(xMat)
	returnMat = zeros((numIt,n))
	ws    = zeros((n,1)); 
	wsTest = ws.copy(); 
	wsMax = ws.copy();
	for i in range(numIt):
		print(ws.T)
		lowestError = inf;
		for j in range(n):
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps*sign
				yTest  = xMat*wsTest
				rssE   = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T
	return returnMat

'''
	购物信息的获取函数
'''
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
	sleep(10)
	myAPIstr = 'get from code.google.com'
	searchURL = 'https://www.googleapis.com/shopping.search/v1/public\
		products?\
		  key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr, setNum)
	pg = request.urlopen(searchURL)
	retDict = json.loads(pg.read())
	for i in range(len(retDict['items'])):
		try:
			currItem = retDict['items'][i]
			if currItem['product']['condition'] == 'new':
				newFlag = 1
			else:
				newFlag = 0
			listOfInv = currItem['product']['inventories']
			for iten in listOfInv:
				sellingPrice = item['price']
				if sellingPrice < origPrc*0.5:
					print("%d\t%d\t%d\t%f\t%f"%\
						(yr, numPce, newFlag, origPrc,sellingPrice))
					retX.append([yr,numPce, newFlag, orgPrc])
					retY.append(sellingPrice)

		except:
			print("problem with item %d"%i)
def setDataCollect(retX,retY):
	searchForSet(retX, retY, 8288,2006,800, 49.99)
	searchForSet(retX, retY, 10030,2002,3096, 269.99)
	searchForSet(retX, retY, 10179,2007,5195, 499.99)



if __name__ == '__main__':
	#loadDataSet('abalone.txt')
	#showLine('ex0.txt',standRegres,k = 1.0)
	#showLine('ex0.txt',)
	abX,abY = loadDataSet('abalone.txt')
	#print(shape(abX)[0])
	#print(shape(abX)[1])
	#abx:4177*8
	#abY:4177*1
	#print(shape(abY)[0])
	#ridgeWeights = ridgeTest(abX,abY)
	#print(ridgeWeights)
	'''
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(221)
	ax.plot(ridgeWeights)
	#plt.show()
	ax1 = fig.add_subplot(222)
	ax1.plot(ridgeWeights)
	plt.show()
	'''
	#ridgeWrights:30*8
	#print(shape(ridgeWeights)[1])
	xArr, yArr = loadDataSet('abalone.txt')
	#stageWise(xArr, yArr, 0.01, 200)
	#ridgeWeights=stageWise(xArr,yArr, 0.001, 5000)
	'''
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	plt.show()
	'''		
	lgx = []; lgy =[];
	setDataCollect(lgx,lgy)
