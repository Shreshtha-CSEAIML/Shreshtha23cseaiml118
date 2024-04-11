import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
train_path = '/kaggle/input/titanic/train.csv'
test_path = '/kaggle/input/titanic/test.csv'
submit_path = '/kaggle/working/submission.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(train_df.head())
print(test_df.head())

print('Label')
print(train_df["Survived"].head())

train_label_equals_1_num = (train_df["Survived"]==1).sum()
print(f'Label == 1 : {train_label_equals_1_num} | ratio : {train_label_equals_1_num / train_df.shape[0]}')keys = train_df.keys()
print("Train DataFrame Missing Value")
print(train_df.isnull().sum())
print("Test DataFrame Missing Value")
print(test_df.isnull().sum())

print(train_df['Embarked'].value_counts())

train_mean_values = {'Age': train_df['Age'].mean(),
                     'Embarked': train_df['Embarked'].value_counts().index[0]}

test_mean_values = {'Age': test_df['Age'].mean(),
                    'Embarked': test_df['Embarked'].value_counts().index[0],
                    'Fare': test_df['Fare'].mean()}

print(f'train_mean_values {train_mean_values}')
print(f'test_mean_values {test_mean_values}')

train_df = train_df.fillna(value=train_mean_values)
test_df = test_df.fillna(value=test_mean_values)


print("Train DataFrame NaN Value")
print(pd.isna(train_df).sum())
print("Test DataFrame NaN Value")
print(pd.isna(test_df).sum())

print("Train DataFrame Missing Value")
print(train_df.isnull().sum())
print("Test DataFrame Missing Value")
print(test_df.isnull().sum())
del train_df['Cabin']
del test_df['Cabin']
keys = train_df.keys()
print("Train DataFrame Missing Value")
print(train_df.isnull().sum())
print("Test DataFrame Missing Value")
print(test_df.isnull().sum())

print(train_df['Embarked'].value_counts())

train_mean_values = {'Age': train_df['Age'].mean(),
                     'Embarked': train_df['Embarked'].value_counts().index[0]}

test_mean_values = {'Age': test_df['Age'].mean(),
                    'Embarked': test_df['Embarked'].value_counts().index[0],
                    'Fare': test_df['Fare'].mean()}

print(f'train_mean_values {train_mean_values}')
print(f'test_mean_values {test_mean_values}')

train_df = train_df.fillna(value=train_mean_values)
test_df = test_df.fillna(value=test_mean_values)
print("Train DataFrame NaN Value")
print(pd.isna(train_df).sum())
print("Test DataFrame NaN Value")
print(pd.isna(test_df).sum())

print("Train DataFrame Missing Value")
print(train_df.isnull().sum())
print("Test DataFrame Missing Value")
print(test_df.isnull().sum())
y = train_df["Survived"]
features_d = ["Pclass", "Sex", "SibSp", "Parch"]
X_d = pd.get_dummies(train_df[features_d], dtype=int)
X_test_d = pd.get_dummies(test_df[features_d], dtype=int)
print(X_d.head())

features_c = ["Age", "Fare"]
X_c = train_df[features_c]
X_test_c = test_df[features_c]
print(X_c.head())

X = pd.merge(X_d, X_c, how='inner', left_index=True, right_index=True)
X_test = pd.merge(X_test_d, X_test_c, how='inner', left_index=True, right_index=True)
print(X.head())
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# xgb_classifier = XGBClassifier()
# xgb_classifier.fit(X_train, y_train)
# train_prediect = xgb_classifier.predict(X_train)
# val_prediect = xgb_classifier.predict(X_val)
# train_acc = accuracy_score(y_train, train_prediect)
# val_acc = accuracy_score(y_val, val_prediect)
# print(f"Train Acc | {train_acc}")
# print(f"Val Acc | {val_acc}")
