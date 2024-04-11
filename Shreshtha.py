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
