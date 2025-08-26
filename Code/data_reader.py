import pandas as pd

train_dataset= pd.read_csv(r'data_files/train.csv')

#print(train_dataset)
#print(train_dataset['id'])

total_missing_values_per_column=train_dataset.isnull().sum().sum()
print(total_missing_values_per_column)
dffdf