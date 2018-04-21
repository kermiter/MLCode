from numpy import *

'''
转换数据
'''
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)#每行映射为浮点数
		dataMat.append(fltLine)
	return dataMat

def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0], :]

	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :]
	#print("mat1",mat1)

	return mat0,mat1
#负责生成叶节点
def regLeaf(dataSet):
	return mean(dataSet[:,-1])
#返回总方差
def regErr(dataSet):
	return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr,ops =(1,4)):
	#tolS:容许的误差下降值,tolN:切分的最少样本值
	tolS = ops[0]; tolN = ops[1];
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:,featIndex]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if ((shape(mat0)[0] < tolN) or (shape(mat1)[1]) < tolN): 
				continue;
				newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS     = newS
	if (S - bests) < tolS:
		return None, leafType(dataSet)
		
	return bestIndex,bestValue

	#递归创建会回归树
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops =(1,4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None : 
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left']  = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def isTree(obj):
	return (type(obj).__name__ == 'dict')

#求树的均值
def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
	if shape(testData)[0] == 0: 
		return getMean(tree)
	if (isTree(tree['right'])) or (isTree(tree['left'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'],
				     tree['spVal'])
		if isTree(tree['left']):
			tree['left'] = prune(tree['left'], lSet)
		if isTree(tree['right']):
			tree['right'] = prune(tree['right'], rSet)
		if not isTree(tree['left']) and not isTree(tree['right']):
			lSet, rSet = binSplitDataSet(testData, tree['spInd'],
						 tree['spVal'])
			errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + \
						sum(power(rSet[:,-1] - tree['right'],2))
			treeMean = (tree['left'] + tree['right']) /2.0
			errorMerge = sum(power(testData[:,-1] - treeMean,2))
			if errorMerge < errorNoMerge:
				print("merging")
				return treeMean
			else:
				return tree
		else:
			return tree
#求解线性模型参数
def linearSolve(dataSet):
	m,n = shape(dataSet)
	X   = mat(ones(m,n)); Y = mat(ones((m,1)))
	X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
	xTx = X.T*X
	if linalg.det(xTx) == 0.0:
		raise NameError('This matrix is singular , cannot do inverse\n\
		try increaseing the second value of ops')
	ws =  xTx.I *(X.T * Y)
	return ws, X,Y

#求解模型的参数
def modelLeaf(dataSet):
	ws, X,Y= linearSolve(dataSet)
	return ws

#计算模型与实际值得差值
def modelErr(dataSet):
	ws, X,Y = linearSolve(dataSet)
	yHat = X*ws
	return sum(power(Y-yHat,2))

'''
用树回归进行预测
'''
def regTreeEval(model, inDat):
	return float(model)

#预测值
def modelTreeEval(model, inDat):
	n = shape(inDat)[1]
	X = mat(ones((1,n+1)))
	X[:,1:n+1] = inDat
	return float(X*model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
	if not isTree(tree):
		return modelEval(tree, inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'], inData)
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'],inData, modelEval)
		else:
			return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
	m = len(testData)
	yHat = mat(zeros((m,1)))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
	return yHat

		return modelEval


	passdef




			
if __name__ == '__main__':
	#testMat = mat(eye(4))
	#print(testMat)
	#mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
	#print("mat1:",mat1)
	myDat = loadDataSet('ex00.txt')
	myMat = mat(myDat)
	createTree(myMat)
	