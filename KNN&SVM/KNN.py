from turtle import distance
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Network_Ads.csv')

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data.iloc[:, 2:3] = imp.fit(data.iloc[:, 2:3]).transform(data.iloc[:, 2:3])

X = data.iloc[:, 0:3]
X = preprocessing.StandardScaler().fit_transform(X)
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = 5
clf = neighbors.KNeighborsClassifier(n_neighbors=k, p=2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Độ chính xác : %.2f %%" % (100 * accuracy_score(y_test, y_pred)))