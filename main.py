import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
Id=test_df['Id']
train_df.drop(columns=["PoolQC", "MiscFeature", "Alley"], inplace=True)
test_df.drop(columns=["PoolQC", "MiscFeature", "Alley"], inplace=True)
num_cols = train_df.select_dtypes(include=["int64", "float64"]).drop(columns=["Id", "SalePrice"], axis=1).columns
cat_cols = train_df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for feature in num_cols:
    train_df[feature].fillna(train_df[feature].median(), inplace=True)
    test_df[feature].fillna(test_df[feature].median(), inplace=True)
for feature in cat_cols:
    train_df[feature].fillna(train_df[feature].mode()[0], inplace=True)
    test_df[feature].fillna(test_df[feature].mode()[0], inplace=True)
    train_df[feature] = le.fit_transform(train_df[feature].values)
    test_df[feature] = le.transform(test_df[feature].values)
x = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=42, test_size=0.2)

model=ensemble.GradientBoostingRegressor(random_state=42,n_estimators=225,min_samples_split=6)
model.fit(train_x, train_y)
val_pred = model.predict(val_x)
print(mean_absolute_error(val_y, val_pred))
model.fit(x,y)
test_pred=model.predict(test_df)
submission=pd.DataFrame(test_pred,columns=['SalePrice'])
Id=pd.DataFrame(Id)
Id=Id.join(submission,)
Id.to_csv(r'C:\Users\alesha\Desktop\SalePrice.csv',index=False)
