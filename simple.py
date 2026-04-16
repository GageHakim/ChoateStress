import pandas as pd
data=pd.read_csv('Data.csv')
data.columns=['Timestamps','Screen', 'Sleep', 'Major_Assessments', 'Hard_Classes', 'HoursHW', 'Form', 'Stress']
print(data.columns)
TP = 0
TN = 0
FP = 0
FN = 0
for index, row in data.iterrows():
    # Logic for Form 6 or 3
    if row['Form'] in [6, 3]:
        if row['Stress'] in ['Low', 'Medium']:
            TN += 1
        else:
            FN += 1
    else:
        if row['Hard_Classes'] >= 3 or row['Sleep'] <= 6:
            if row['Stress'] == 'High':
                TP += 1
            else:
                FP += 1
print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")
print(f"Accuracy: {round(((TP+TN)/(TP+TN+FP+FN))*100)}%")