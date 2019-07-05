import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("Salary_Data.csv")
x=ds.iloc[:,:-1].values
y=ds.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.title("salary vs no of exprience")
plt.xlabel("exprience")
plt.ylabel("salary")
#verify on test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.title("salary vs no of exprience")
plt.xlabel("exprience")
plt.ylabel("salary")
