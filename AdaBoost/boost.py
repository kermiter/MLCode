from numpy import *
from adaboost import *

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'It':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray
#单层决策树生成函数
def buildStump(dataArr, classLabels, D):
	
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	numSteps = 10.0; 
	bestStump = {};
	bestClassEst = mat(zeros((m,1)))
	minError = inf
	for i in range(n):
		rangeMin = dataMatrix[:,i].min();
		rangeMax = dataMatrix[:,i].max();
		stepSize = (rangeMax - rangeMin)/numSteps
		cnt = 0
		for j in range(-1,int(numSteps)+1):
			for inequal in ['It','gt']:
				threshVal = (rangeMin + float(j)*stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[labelMat == predictedVals] = 0
				weightedError = D.T*errArr
				'''
				print("split: dim %d, thresh %.2f, thresh inequal: \
					%s, the weighed error is %.3f" %\
					(i, threshVal, inequal, weightedError))
				#print('cnt:%d' %cnt);
				cnt = cnt+1
				'''
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal

	return bestStump, minError, bestClassEst
	
if __name__ == '__main__':
	datMat, classLabels = loadSimpData();
	#print(datMat)
	'''
	row = shape(datMat)[0];
	D = mat(ones((row,1))/row);
	bestStump, minError, bestClassEst = buildStump(datMat,classLabels,D);
	print(bestStump)
	print(minError)
	print(bestClassEst)
	'''		
