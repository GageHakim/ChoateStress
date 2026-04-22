import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('Data.csv')
df.columns=['Timestamps','Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form', 'Stress']

df['Stress']=np.where(df['Stress'].isin(['Low', 'Medium']), 0, 1)

features=['Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form']
X=df[features]
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
y=df['Stress']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=LinearRegression()
model.fit(X_scaled, y)

predictions=model.predict(X_scaled)

binary_prediction=(predictions>=.5).astype(int)


accuracy=accuracy_score(y, binary_prediction)
print(accuracy)

cm=confusion_matrix(y, binary_prediction)
print('Confusion Matrix:')
print(cm)
print(features)
print(model.coef_)
print(model.intercept_)