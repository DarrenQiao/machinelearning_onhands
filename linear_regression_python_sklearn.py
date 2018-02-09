import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  linear_model

#读数据
data = pd.read_csv('C://Users//Administrator//data_for_linearregression.txt', names = ['population', 'gdp'])

#转换成x和y的矩阵
x = data.as_matrix(columns=['population'])
y = data.as_matrix(columns=['gdp'])

#调用线性回归函数直接拟合数据
reg = linear_model.LinearRegression()
reg.fit(x, y)
predict_y = reg.predict(x)

#画出拟合结果
plt.scatter(x, y)
plt.plot(x, predict_y, 'r')
plt.xlabel("population/(10,000)")
plt.ylabel("profit/(10,000)")
plt.show()