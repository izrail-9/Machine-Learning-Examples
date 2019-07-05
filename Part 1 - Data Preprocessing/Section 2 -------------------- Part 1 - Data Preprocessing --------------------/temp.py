import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,-1].values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x[:,0]=le.fit_transform(x[:,0])
ohe=OneHotEncoder(categorical_features=[0])
x=ohe.fit_transform(x).toarray()

