import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

#Load and check for null values in Train dataset
train_dataset= pd.read_csv(r'data_files/train.csv')
test_dataset =pd.read_csv(r'data_files/test.csv')

# print(train_dataset.head())
# print(train_dataset.columns)

#check for missing values in an column of the train dataset
total_missing_values_per_column=train_dataset.isnull().sum()
print(total_missing_values_per_column)

# Grouping the age in categories
train_dataset['age']= pd.cut(train_dataset['age'], bins=[17, 25, 40, 60, 95], labels = ['child', 'adult', 'mid-age', 'senior'])
test_dataset['age']=pd.cut(test_dataset['age'], bins=[17, 25, 40, 60, 95], labels = ['child', 'adult', 'mid-age', 'senior'])
print(train_dataset.head())

#job column check 
print(train_dataset['job'].unique()) #there is a label called unknown and i have to investigate it
print((train_dataset['job']=='unknown').sum())

#check for the most frequent job in the train dataset
print(train_dataset['job'].mode())

#Replace the unknown job possitions with the most frequent
train_dataset['job']=train_dataset['job'].replace("unknown", train_dataset['job'].mode()[0])
test_dataset['job']=test_dataset['job'].replace("unknown", test_dataset['job'].mode()[0])
print(train_dataset['job'].unique())

#search for other data that need manipulation

#Separate the data into categorical, numerical and ordinal values
X = train_dataset.drop(columns=["y", "id"])  # drop target and id
y = train_dataset["y"]

test_ids = test_dataset["id"]
X_test = test_dataset.drop(columns=["id"])

print(train_dataset.dtypes)
numeric_columns=["balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_columns = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
ordinal_columns = ["age"]

numeric_df=pd.DataFrame(train_dataset, columns=numeric_columns)
categorical_df=pd.DataFrame(train_dataset,columns=categorical_columns)
ordinal_df=pd.DataFrame(train_dataset, columns=ordinal_columns)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns + ordinal_columns)
    ]
)

#Processor for train dataset
X_processed = preprocessor.fit_transform(X)

# Get column names
ohe = preprocessor.named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical_columns + ordinal_columns)
all_features = numeric_columns + list(ohe_features)

# Wrap into a DataFrame the train dataset
X_df = pd.DataFrame(X_processed, columns=all_features, index=X.index)

#transform test dataset
X__test_processed = preprocessor.transform(X_test)
X_test_df = pd.DataFrame(X__test_processed, columns=all_features, index=X_test.index)

print("Transformed features shape:", X_df.shape)
print(X_df.head())


# Logistic Regression in train data set 

model = LogisticRegression()
model.fit(X_df, y)
y_pred = model.predict(X_df)
y_pred_proba = model.predict_proba(X_df)

print("Training Accuracy:", accuracy_score(y, y_pred))
print("Training AUC:", roc_auc_score(y, y_pred_proba[:, 1]))


cv_scores = cross_val_score(
    model, 
    X_df, 
    y, 
    cv=5, 
    scoring="roc_auc"
)

print("ROC AUC scores from 5 folds:", cv_scores)
print("Mean ROC AUC:", cv_scores.mean())
print("Std ROC AUC:", cv_scores.std())


#prediction on test dataset
y_pred_proba_test = model.predict_proba(X_test_df)[:, 1]

# Build submission file
submission = pd.DataFrame({
    "id": test_ids,
    "y": y_pred_proba_test
})

# submission.to_csv("results/submission.csv", index=False)
# print("✅ Submission file created")
# print(submission.head())





# Initialize Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)  # you can tune max_depth

# Train on training data
dt_model.fit(X_df, y)

# Predictions on training data
y_pred = dt_model.predict(X_df)
y_pred_proba = dt_model.predict_proba(X_df)[:, 1]

# Training performance
print("Training Accuracy:", accuracy_score(y, y_pred))
print("Training ROC AUC:", roc_auc_score(y, y_pred_proba))

# Cross-validation
cv_scores = cross_val_score(
    dt_model,
    X_df,
    y,
    cv=5,
    scoring="roc_auc"
)
print("ROC AUC scores from 5 folds:", cv_scores)
print("Mean ROC AUC:", cv_scores.mean())
print("Std ROC AUC:", cv_scores.std())

# Predictions on test dataset
y_pred_proba_test = dt_model.predict_proba(X_test_df)[:, 1]

# Build submission file
submission = pd.DataFrame({
    "id": test_ids,
    "y": y_pred_proba_test
})
submission.to_csv("results/submission_decision_tree.csv", index=False)
print("Submission file created")
print(submission.head())