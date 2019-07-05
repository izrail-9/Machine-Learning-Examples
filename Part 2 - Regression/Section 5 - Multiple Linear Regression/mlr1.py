import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("50_Startups.csv")
x=ds.iloc[:,0:4].values
y=ds.iloc[:,4:5].values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
le=le.fit(x[:,3])
x[:,3]=le.transform(x[:,3])
ohe=OneHotEncoder(categorical_features=[3])
x=ohe.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/5)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


