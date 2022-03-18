import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = pd.read_csv('data_Logistic.csv')

x_train, x_test, y_train, y_test = train_test_split(df[['Age']], df.Purchased, train_size=0.2)


model = LogisticRegression()

model.fit(x_train, y_train)

y_hat = model.predict(x_test)

print('\nThe mean accuracy: ', model.score(x_test,y_test))
print('Classification Report:\n', classification_report(y_test, y_hat))

