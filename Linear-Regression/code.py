# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here

##loading the dataset
df = pd.read_csv(path)

## analyzing the first 5 rows
print(df.head())

##create the feature set
X = df.drop('list_price',axis=1)

##create the target set
y = df['list_price']

## splitting into train and test set using test_train_split
X_train, y_train, X_test, y_test = train_test_split(X,y,random_state=6, test_size=0.3)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        

##getting the list of columns in the train set
cols = X_train.columns

## initializing the subplots with 3 rows and 3 cols
fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(15,10))

# ## looping through the rows of the subplot
for i,row in enumerate(axes):
    ##looping through the subplot one by one
    for j,ax in enumerate(row):
        ##added to pick appropriate columns from train set
        col = cols[i*3 + j]
        ## plotting the scatter plot for each of the feature
        ax.scatter(X_train[col],y_train)
        ##setting x label
        ax.set_xlabel(col)
        ##setting y label
        ax.set_ylabel('list_price')
# code ends here


# --------------
# Code starts here

##find correlation between features
corr = X_train.corr()

print(corr)

## identify features having correlation higher than 0.75
high_corr_features = corr[(corr>0.75) | (corr <-0.75)]

##remove highly correlated features from train set
X_train.drop(columns=['val_star_rating','play_star_rating'], inplace=True)

##remove highly correlated features from test set
X_test.drop(columns=['val_star_rating','play_star_rating'], inplace=True)


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

##instantiate the Linear Regression Model
regressor = LinearRegression()

##fit the model
regressor.fit(X_train, y_train)

##make predictions from the model
y_pred = regressor.predict(X_test)

##calculate the mean squared error
# mse = mean_squared_error(y_pred,y_test)

mse = sum((y_pred-y_test)**2)/y_pred.shape[0]

print(mse)

##calculate the R squared error
r2 = r2_score(y_test, y_pred)

print(r2)
# Code ends here


# --------------
# Code starts here

##calculate the residual for the predicted values
residual = y_test - y_pred

##plot the histogram of the residual error
plt.hist(residual)
plt.xlabel('Residual Error')
plt.ylabel('Frequency')
plt.title('Frequency of residual error')


# Code ends here


