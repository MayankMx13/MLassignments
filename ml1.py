
import pandas as pd

df = pd.read_csv('/content/train_e26362d54a8cfdba92a437365ae76635.csv')



df.head()



df= df.fillna(df.mean())


X = df.drop(['LotArea'], axis=1)

y = df['LotArea']



categorical_cols = ['LotArea', 'SaleCondition', 'LandContour']

X = pd.get_dummies(X, columns=categorical_cols)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)



from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)



print(f"Mean Squared Error: {mse}")

print(f"R-squared: {r2}")

print(df.columns)
