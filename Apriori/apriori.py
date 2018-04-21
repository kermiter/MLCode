from time import sleep
#from votesmart import votesmart
#该模块需要下载

#玩具数据集
def loadDataSet():
	return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

#检测候选项集是否tran的子集
def createC1(dataSet):
	C1 = []

	for transaction in  dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return list(map(frozenset, C1))
'''
	D:数据集
	Ck: 候选项集列表
	minSupport：支持度
	caution: Python3中map对象不可直接迭代，需要转换为list
'''
def scanD(D, Ck, minSupport):
	ssCnt = {}
	numItems = len(list(D))
	#print(list(D))
	#print(numItems)
	#print("Ck",list(Ck))
	#print(D)
	for tid in D:
		#print("tid",tid)	
		for can in Ck:
			if can.issubset(tid):
				if can not in ssCnt.keys(): 
					ssCnt[can] = 1
					#print(ssCnt.keys())
				else: 
					ssCnt[can] += 1
	retList=[]
	supportData = {}
	#print("numItems",numItems)
	#print("ssCnt",ssCnt)
	for key in ssCnt.keys():
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0,key)
		supportData[key]  = support
	return retList, supportData
#####以下函数还未仔细深究#######
#由Lk->Ck
def aprioriGen(Lk,k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2];
			L2 = list(Lk[j])[:k-2];
			L1.sort();L2.sort();
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])
	return retList
#生成所有支撑度>0.5的列表和相应的支持度
def apriori(dataSet, minSupport = 0.5):
	C1 = createC1(dataSet)
	D  = list(map(set, dataSet))
	#print("C1:",C1)
	#print("D:",D)
	L1, supportData = scanD(D, C1, minSupport)
	L  = [L1]
	#print(type(L1))
	k  = 2
	while (len(L[k-2])>0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData
	

def generateRules(L, supportData, minConf = 0.7):
	bigRuleList = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if (i>1):
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >= minConf:
			print(freqSet-conseq,'-->',conseq,'conf:',conf)
			br1.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
	m = len(H[0])
	if (len(freqSet) > (m+1)):
		Hmp1 = aprioriGen(H, m+1)
		Hmp1 = calcConf(freqSet,  Hmp1, supportData, br1, minConf)
		if (len(Hmp1) > 1):
			rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)
'''			
#该函数依赖votesmart模块
def getActionIds():

	actionIdList = []; billTitleList = []
	fr = open('recent20bills.txt')
	for line in fr.readlines():
		billNum = int(line.split('\t')[0])
		try:
			billDetail = votesmart.votes.getBill(billNum)
			for action in billDetail.actions:
				if action.level == 'House'and \
					(action.stage == 'Passage' or \
						action.stage == 'Amendment Vote'):
					print("bill:%d has actionId:%d" % (billNum, actionId))
					billTitleList.append(line.strip().split('\t')[1])
		except:
			print("problem getting bill %d" %billNum)
		sleep(1)
	return actionIdList，billTitleList


def getTransList(actionIdList, billTitleList):
	itemMeaning = ['Republican', 'Democratic']
	for billTitle in billTitleList:
		itemMeaning.append('%s -- Nay' % billTitle)
		itemMeaning.append('%s -- Yea' % billTitle)
	transDict = {}
	voteCount = 2
	for actionId in actionIdList:
		sleep(3)
		print("getting votes for actionId:%d" %actionId)
		try:
			voteList = votesmart.votes.getBillActionVotes(actionId)
			for vote in voteList:
				if not transDict.has_key(vote.candidateName):
					transDict[vote.candidateName] = []
					if vote.officeParties == 'Democratic':
						transDict[vote.candidateName].append(1)
					elif vote.officeParties == 'Republican':
						transDict[vote.candidateName].append(0)
				if vote.action == 'Nay':
					transDict[vote.candidateName].append(voteCount)
				elif vote.action == 'Yea':
					transDict[vote.candidateName].append(voteCount+1)
		except:
			print("problem getting actionId:%d" %acitonId)
		voteCount += 2
	return transDict, itemMeaning
'''


					
						

if __name__ == "__main__":
	dataSet = loadDataSet()
	C1      =createC1(dataSet)
	#print(type(C1))
	D       = list(map(set, dataSet))
	L1, supportData = scanD(D,C1,0.5)
	#print(L1)
	L, suppData = apriori(dataSet,minSupport =0.5)
	#print("L:",L)
	#print("L[0]:",L[0])
	#print("L[1]:",L[1])
	#print("L[2]:",L[2])
	#print("L[3]:",L[3])
	#C2 = aprioriGen(L[0],2)
	#print("C2",C2)
	rules = generateRules(L,suppData,minConf =0.7)
	print("rules:",rules)