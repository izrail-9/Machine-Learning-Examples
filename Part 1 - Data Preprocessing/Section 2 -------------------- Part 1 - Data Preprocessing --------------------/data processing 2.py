import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("Data2.csv")
x=ds.iloc[:,:7].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
#x[:,0]=le.fit_transform(x[:,0])
#x[:,3]=le.fit_transform(x[:,3])
#ohe=OneHotEncoder(categorical_features=[0])
#x=ohe.fit_transform(x).toarray()
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
imp=imp.fit(x[:,6:7])
x[:,6:7]=imp.transform(x[:,6:7])