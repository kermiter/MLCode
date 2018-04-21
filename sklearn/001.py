
###matplot.version = 2.0.2
###sklearn.version = 0.19
import matplotlib.pyplot as plt 
import matplotlib
#from matplotlib.font_manager import FonProperties
from  sklearn.linear_model import LinearRegression
import sklearn 
def runplt():
	plt.figure()
	plt.title('披萨价格与直径')
	plt.xlabel('直径(英寸)')
	plt.ylabel('价格(美元)')
	plt.axis([0,25,0,25])
	plt.grid(True)
	return plt

#创建模型

x     = [[6],[8],[10],[14],[18]]
y     = [[7],[9],[13],[17.5],[18]]
plt   = runplt()
plt.plot(x,y,'k.')
plt.show()
model = LinearRegression()
model.fit(x,y)
print('预测一张12英寸披萨价格：$%0.2f'%model.predict([[12]])[0])
print(matplotlib.__version__)

