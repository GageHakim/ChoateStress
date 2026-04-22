# Not any part, wanted to experiment and it did not really work
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score
model=LogisticRegression()

df=pd.read_csv('Data.csv')
df.columns=['Timestamps','Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form', 'Stress']

df['Stress']=np.where(df['Stress'].isin(['Low', 'Medium']), 0, 1)
features=['Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form']
X=df[features]
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

X_scaled['parabolic_form']=(X_scaled['Form']-0.5)**2
print(X_scaled)
features=['Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'parabolic_form']
df.drop(columns='Form')
y=df['Stress']

model.fit(X_scaled, y)
predictions_prob=model.predict_proba(X_scaled)[:, 1]
predictions=model.predict(X_scaled)
print(f"Accuracy: {accuracy_score(y, predictions)}")
print(f"Log Loss: {log_loss(y, predictions_prob)}")
print(features)
print(model.coef_)
print(confusion_matrix(predictions, y))
print(roc_auc_score(y, predictions_prob))