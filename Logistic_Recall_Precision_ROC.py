# Part 5 Code
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, roc_auc_score
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

#Precision Recall Graph, not sure why I did it but it came up in documentation so yeah.
import matplotlib.pyplot as plt

precision, recall, threshold=precision_recall_curve(y, predictions_prob)
display_precision_recall = PrecisionRecallDisplay(precision=precision, recall=recall)
print(predictions_prob)
print(threshold)


display_precision_recall.plot()
plt.show()

#ROC AUC
FPR, TPR, thresholds_ROC=roc_curve(y, predictions_prob)
display_ROC_AUC = RocCurveDisplay(tpr=TPR, fpr=FPR)

display_ROC_AUC.plot()
plt.show()

#Now lets calculate F1. Since I already have precision and recall from the code above so I guess my life is extra easy.

#This code just chooses the non 0 and 100 septiles I guess of the thresholds chosen in the cocde
#So 5 thresholds are chosen, and the thresholds are pretty much just the predicted probabilties of each value. And hopefully this makes sense since a threshold only matters when a number is greater than it.
chosen_thresholds=[]
len_predictions=len(threshold)
for i in range(1,7):
    chosen_thresholds.append(int((len_predictions*i)/7))
print(chosen_thresholds)

for thresh in chosen_thresholds:
    print(f"The threshold chosen is {threshold[thresh]}.\n Its Precision is {precision[thresh]} \n Its Recall is {recall[thresh]} \n Its F1 Score is {(2*precision[thresh]*recall[thresh])/(precision[thresh]+recall[thresh])}")

print(roc_auc_score(y, predictions_prob))