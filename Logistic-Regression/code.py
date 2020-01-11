# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
# load the file using pandas
df = pd.read_csv(path)

# observe the top rows of the dataset
print(df.head())

# split the data into features and labels
X = df.drop(columns=['insuranceclaim'], axis=1)
y = df['insuranceclaim']

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

#print the shapes of train and test set
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
# plotting box plot for train features
sns.boxplot(y='bmi', data=X_train)

#Set quantile equal to 0.95for X_train['bmi'
q_value = X_train['bmi'].quantile(q=0.95)

#Check the value counts of the y_train
print("Distribution of labels in y_train is:",y_train.value_counts())
# Code ends here


# --------------
# Code starts here

#find the correlation between features
relation = X_train.corr()

print(relation)

#plot the pairplot between all the features of the train set
sns.pairplot(X_train)

# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
# identifying the columns for countplot
cols = ['children','sex','region','smoker']

#defining the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

# looping through the rows and cols
for i,row in enumerate(axes):
    for j,ax  in enumerate(row):
        # plotting countplot
        col = cols[i*2 + j] 
        sns.countplot(x=col, data=X_train , hue= y_train, ax=axes[i,j])


# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
# initialize the Logistic regression
lr = LogisticRegression(random_state=9)

# Using GridSearchCV do exhaustive search over specified parameter values for an estimator
grid = GridSearchCV(estimator=lr, param_grid = parameters)
# fit the model to train data
grid.fit(X_train, y_train)

# do the predictions
y_pred = grid.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve

# Code starts here
# calculate the roc-auc score
score = roc_auc_score(y_test, y_pred)

print('roc_auc_score is:',score)

# Calculate the probability using grid search with the best found parameters
y_pred_proba = grid.predict_proba(X_test)[:,1]

# calculate the false positive recall and true positive recall using roc curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# #Calculate the roc_auc score of 
roc_auc = roc_auc_score(y_test, y_pred_proba)

# # plot the auc curve
plt.plot(fpr, tpr,label='Logistic Model, auc'+ str(roc_auc))
# book-keeping
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Code ends here


