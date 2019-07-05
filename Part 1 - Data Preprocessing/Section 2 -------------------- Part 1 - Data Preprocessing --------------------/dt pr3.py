import numpy as np
import pandas as pd
ds=pd.read_csv("Data.csv")
x=ds.iloc[:,0:3].values
y=ds.iloc[:,3].values
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="median")
imp=imp.fit(x[:,1:3])
x[:,1:3]=imp.transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lec=LabelEncoder()
lec=lec.fit(x[:,0])
x[:,0]=lec.transform(x[:,0])
ohe=OneHotEncoder(categorical_features=[0])
x=ohe.fit_transform(x).toarray()
lec=lec.fit(y)
y=lec.transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


