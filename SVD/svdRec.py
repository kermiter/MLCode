from numpy import *
from numpy import linalg as la

def loadExData():
	return  [[1, 1, 1, 0, 0],
			 [2, 2, 2, 0, 0],
			 [1, 1, 1, 0, 0],
			 [5, 5, 5, 0, 0],
			 [1, 1, 0, 2, 2],
			 [0, 0, 0, 3, 3],
			 [0, 0, 0, 1, 1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def function():
	pass

#根据欧式距离得出的相似度
def euclidSum(inA, inB):
	return 1.0/(1.0+la.norm(inA-inB))

#皮尔逊系数
def pearsSim(inA, inB):
	if len(inA)  <3: return 1.0
	return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]#什么？

#cos相似度
def cosSim(inA, inB):
	num = float(inA.T*inB)
	#print("inA",inA)
	#print("inB",inB)
	#print("num",num)
	denom = la.norm(inA)*la.norm(inB)
	return 0.5+0.5*(num/denom)
#不做奇异值分解的计算相似度的函数
def standEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]
	simTotal = 0.0; ratSimTotal = 0.0
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0: 
			continue
		overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
		if len(overLap) == 0: 
			similarity = 0
		else:
			similarity = simMeas(dataMat[overLap,item],\
								 dataMat[overLap, j])
		simTotal += similarity
		ratSimTotal += similarity*userRating
	if simTotal ==0:
		return 0
	else:
		return ratSimTotal/simTotal

def svdEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]
	simTotal = 0.0; ratSimTotal = 0.0;
	U,Sigma,VT = linalg.svd(dataMat)#奇异值分解
	Sig4 = mat(eye(4)*Sigma[:4])#eye:创建单位矩阵
	#print(eys(4))
	#print("Sigma[:4],type(sigma)",Sigma[:4])
	#print("Sig4",Sig4)
	xformedItems = dataMat.T * U[:,:4] *Sig4;#将数据转换到低维空间
	#print(type(dataMat[0,0]))
	#print(xformedItems)
	for j in range(n):
		userRating = dataMat[user,j]
		if userRating ==0 or j == item:
			continue
		similarity = simMeas(xformedItems[item,:].T,\
							 xformedItems[j,:].T)
		print('the %d and %d similarity is: %f'%(item,j,similarity))
		print("the %d and %d similarity is: %f"%(item,j,similarity))
		simTotal += similarity
		ratSimTotal += similarity*userRating
		if simTotal == 0:
			return 0
		else:
			return ratSimTotal/simTotal



#

def recommend(dataMat, user, N=3, simMeas=cosSim,estMethod = standEst):
	unratedItems = nonzero(dataMat[user,:].A==0)[1]

	if len(unratedItems)==0:
		print("you rated everything")
	itemScores =[]
	for item in unratedItems:
		estimatedScore = estMethod(dataMat, user, simMeas, item)
		itemScores.append((item, estimatedScore))
	#print(itemScores[0][1])	
	return sorted(itemScores, \
					key=lambda jj: jj[1], reverse= True)[:N]
	#print(len(itemScores))
	




if __name__ == '__main__':
	Data = loadExData2()
	U, Sigma, VT = linalg.svd(Data)
	Sig2 = Sigma**2
	myMat = mat(Data)
	recommend(myMat,1,N=2,estMethod= svdEst)
	#print(Sig2)

'''
	print("U",U)
	print("Sigma",Sigma)
	print("VT",VT)
	myMat = mat(Data)
	#print(myMat)
	myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
	myMat[3,3] = 2
	#print(myMat)
	#print(help(sorted))
	res = recommend(myMat,2)
	print(res)
'''