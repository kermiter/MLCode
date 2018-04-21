'''
	list
	author:ljq
	date: 20171102
'''

#一维
aList = [123, 'abc']
del(aList[0])
del(aList[0])

numlist = ['a', 'd','2']
#多维
mixup_list = [4.0, [1, 'x'], 'beef', -1.9+6j]

#连接两个列表，extend == '+='
mixup_list.extend(numlist)
mixup_list += numlist

#序列类型函数，len,max,min, sorted，reversed，enumerate,zip
albums = ['tales', 'robot', 'pyramid']


if __name__ == '__main__':
	'''print(aList)
	print(mixup_list)
	print(mixup_list[: : -1])
	print(mixup_list[: : -2])
	print(mixup_list[:-1])
	print(mixup_list[:-2])
	help(zip)
	'''
	for  i,album in enumerate(albums):
		print(i,album)