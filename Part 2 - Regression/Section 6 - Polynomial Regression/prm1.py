import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("Position_Salaries.csv")
x=ds.iloc[:,1:2].values
y=ds.iloc[:,2:3].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
y_pred=regressor.predict(x)

'''plt.scatter(x,y,color='red',linewidths="5",edgecolors='green')
plt.plot(x,y_pred,color='green')
plt.title("real vs pred")
plt.xlabel("post")
plt.ylabel("salary")
plt.show()'''

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
regre2=LinearRegression()
regre2.fit(x_poly,y)
y_2=regre2.predict(x_poly)

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red',linewidths="0.1",edgecolors='green')
plt.plot(x_grid,regre2.predict(poly_reg.fit_transform(x_grid)),color='green')
plt.title("real vs pred")
plt.xlabel("post")
plt.ylabel("salary")
plt.show()

