# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# load the data from dataset
df = pd.read_csv(path)

# visualize the first five rows of the dataset
print(df.head())

# split the dataset into features and targets
X = df.drop('Price',axis=1)
y = df['Price']

# split the dataset inot test and train set 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=6)

# identify the correlation between different features of the train set
corr = X_train.corr()

print(corr)

# visualize the correlation between features using heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr,cmap='YlGnBu')

#Code starts here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
#initialize the LinearRegression
regressor = LinearRegression()

# fit the model to train data
regressor.fit(X_train, y_train)

# predict output using the regressor
y_pred = regressor.predict(X_test)

# identify the r2 score of the model
r2 = r2_score(y_test,y_pred)

print(r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
#initialize the lasso model
lasso = Lasso()

# fit the model to train data
lasso.fit(X_train, y_train)

# predict the prices
lasso_pred = lasso.predict(X_test)

# identify the r2 scrore of the predicted value
r2_lasso = r2_score(y_test, lasso_pred)

print(r2_lasso)


# --------------
from sklearn.linear_model import Ridge

# Code starts here

ridge = Ridge()

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

r2_ridge = r2_score(y_test, ridge_pred)

print("value of r2 score for Ridge Regression is: ",r2_ridge)

# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
#initialize the linear regression model
regressor = LinearRegression()

#Cross validation with linear model
score = cross_val_score(regressor,X_train, y_train, cv=10 )

#  identify the mean cross val score
mean_score = np.mean(score)

print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

#initialize the model using make pipeline
model = make_pipeline(PolynomialFeatures(2),LinearRegression())

# fit the model 
model.fit(X_train, y_train)

# predict the test labels
y_pred = model.predict(X_test)

# calculate the r2 score
r2_poly = r2_score(y_test,y_pred)

print('r2 score using polynomial features is:' , r2_poly)


