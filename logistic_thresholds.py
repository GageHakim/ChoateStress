# Part 3 Code
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
model=LogisticRegression()

df=pd.read_csv('Data.csv')
df.columns=['Timestamps','Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form', 'Stress']

df['Stress']=np.where(df['Stress'].isin(['Low', 'Medium']), 0, 1)

features=['Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form']
X=df[features]
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
y=df['Stress']

model.fit(X_scaled, y)
predictions_prob=model.predict_proba(X)[:,1]
print(model.predict_proba(X))
scores=[]

for i in range(0,101):
    threshold=i/100
    predictions=(predictions_prob>threshold).astype(int)
    scores.append(accuracy_score(predictions, y)*100)

import matplotlib.pyplot as plt
print(max(scores), scores.index(max(scores)))
#so accoridng to this the best threshold which yields an accuracy of 77.78%, .42 is the best threshold
thresholds=range(0, 101)

fig, ax= plt.subplots()

ax.plot(thresholds, scores)

ax.set(xlabel='Thresholds (%)', ylabel='Accuracy (%)')
ax.grid()

fig.savefig('Accuracy Graph')
plt.show()

