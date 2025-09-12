import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

#Load and check for null values in Train dataset
train_dataset= pd.read_csv(r'data_files/train.csv')
print(train_dataset.head())
print(train_dataset.columns)

#check for missing values in an column of the train dataset
total_missing_values_per_column=train_dataset.isnull().sum()
print(total_missing_values_per_column)

# Grouping the age in categores
train_dataset['age']= pd.cut(train_dataset['age'], bins=[17, 25, 40, 60, 95], labels = ['child', 'adult', 'mid-age', 'senior'])
print(train_dataset.head())

#job column check 
print(train_dataset['job'].unique()) #there is a label called unknown and i have to investigate it

print((train_dataset['job']=='unknown').sum())

#check for the most frequent job in the train dataset

print(train_dataset['job'].mode())

#Replace the unknown job possitions with the most frequent

train_dataset['job']=train_dataset['job'].replace("unknown", train_dataset['job'].mode()[0])

print(train_dataset['job'].unique())


# columns = ["id","age","job","marital","education","default","balance","housing","loan","contact",
#            "day","month","duration","campaign","pdays","previous","poutcome","y"]

# df = pd.DataFrame(train_dataset, columns=columns)

# # Numerical Features
# Numerical_features = ["age","balance","day","duration","campaign","pdays","previous"]
# corr_matrix=df[Numerical_features].corr(method="pearson")

# # Plot heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
# plt.title("Correlation Heatmap - Numerical Features")
# plt.show()