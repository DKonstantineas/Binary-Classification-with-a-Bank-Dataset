import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

#Load and check for null values in Train dataset
train_dataset= pd.read_csv(r'data_files/train.csv')

#print(train_dataset)
#print(train_dataset['id'])

total_missing_values_per_column=train_dataset.isnull().sum().sum()
#print(total_missing_values_per_column)



#Load and check for null values in Test dataset

test_dataset=pd.read_csv(r'data_files/test.csv')
missing_values_in_test_dataset=test_dataset.isnull().sum().sum()
#print(missing_values_in_test_dataset)


#Heatmap for train_dataset
#print(train_dataset.head().to_string)

columns = ["id","age","job","marital","education","default","balance","housing","loan","contact",
           "day","month","duration","campaign","pdays","previous","poutcome","y"]

df = pd.DataFrame(train_dataset, columns=columns)

# Encode categorical columns
categorical_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome","y"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(df_encoded)

corr_matrix = df_encoded.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()