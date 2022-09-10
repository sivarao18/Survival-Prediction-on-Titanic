import pandas as pd
import numpy as np
import pickle
data=pd.read_csv('Titanic2.csv')
X=data.iloc[:,:8].values
from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,:]=si.fit_transform(X[:,:])
Y=data.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
rfClassifier=RandomForestClassifier(n_estimators=400)
rfClassifier.fit(X_train,Y_train)

Y_rf_pred=rfClassifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_rf_pred,Y_test))

print(rfClassifier.predict([[1,3,1,22,1,0,7.25,2]]))

f=open('model.pkl','wb')
pickle.dump(rfClassifier,f)
f.close()