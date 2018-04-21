import os,sys
import pdb

if __name__ == '__main__':
	a = 'adb shell wm size'
	b = os.popen(a).read()
	pdb.set_trace()
	c = input('请输入：')
	print(c)	


