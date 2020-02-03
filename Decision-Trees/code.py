# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
# load the dataset
data = pd.read_csv(path)

# Split the data into train and test set, dropping customerId and paid back loan for feature set, here paid
# back loan is the target variable
X = data.drop(columns=['customer.id','paid.back.loan'], axis=1)
y = data['paid.back.loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
# getting a value count of the distribution of the target variable
fully_paid = y_train.value_counts()

# plotting the distribution
fully_paid.plot.bar(x=fully_paid, rot=0)
plt.xlabel('Whether the loan paid back')
plt.title('Percent of loan paid back ')


# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
#check the dtypes of columns
print(X_train.dtypes)

# removing the special character from train datasets and converting the datatype to float
X_train['int.rate'] = X_train['int.rate'].str.replace('%','').astype(float)
X_train['int.rate'] = X_train['int.rate']/100

# doing the same preparation for test set
X_test['int.rate'] = X_test['int.rate'].str.replace('%','').astype(float)
X_test['int.rate'] = X_test['int.rate']/100

# subsetting only the numerical columns
num_df = X_train.select_dtypes(include=np.number)
print(num_df.shape)

# subsetting categorical columns
cat_df = X_train.select_dtypes(include='object')
print(cat_df.shape)

# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
# lis of all the numerical columns
cols = num_df.columns

# creating subplot with 9 rows
fig, axes = plt.subplots(nrows=9 , ncols=1, figsize=(10,16))

# loop through the subplots and plot boxplot for all the numerical columns
for i,row in enumerate(axes):
    sns.boxplot(x = y_train , y = num_df[cols[i]], ax=axes[i])

# Code ends here


# --------------
# Code starts here

#list of all categorical columns
cols = cat_df.columns

# create subplot 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

# iterate through the subplots and plot the countplot for all categorical features
for i,row in enumerate(axes):
    for j,ax in enumerate(row):
        col = cols[i*2 + j]
        sns.countplot(x= X_train[col], hue = y_train , ax = axes[i,j])
        plt.xticks(rotation=90)


# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here

# iterate over categorical columns
for col in list(cat_df.columns):
    #initialise Label Encoder
    le = LabelEncoder()
    # fit and transform categorical train features
    X_train[col] = le.fit_transform(X_train[col])
    # transform the test categorical features
    X_test[col] = le.transform(X_test[col])


# check the distribution among train labels
print(y_train.value_counts())

#check the distribution among test labels
print(y_test.value_counts())

# fit and transforming the labels for both train and test
le_l = LabelEncoder()
y_train = pd.Series(le_l.fit_transform(y_train))
y_test = pd.Series(le_l.transform(y_test))

#verifying whether correct assignment has been made by label encoder
print(y_train.value_counts())
print(y_test.value_counts())

# instantiate Decision tree classifier
model = DecisionTreeClassifier(random_state=0)
# fit the training data to model
model.fit(X_train, y_train)
# calculate the accuracy score
acc = model.score(X_test, y_test)

print('Accuracy of the decision tree classifier is:', acc)

# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here

# initialise the DecisionTreeClassifier
model_2 = DecisionTreeClassifier(random_state=0)

# initialise the GridSearchCV object
p_tree = GridSearchCV(estimator= model_2 , param_grid = parameter_grid , cv = 5)

#  fit the model to train set
p_tree.fit(X_train, y_train)

# calculate the accuracy score
acc_2 = p_tree.score(X_test, y_test)

# print the accuracy
print('Accuracy of Decision tree classifier after pruning the tree to 7 depth is:', acc_2)

# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here

# instantiate the export_graphviz object to visualize the tree
dot_data = export_graphviz(decision_tree = p_tree.best_estimator_ ,out_file=None, feature_names= X.columns,filled= True, class_names = ['loan_paid_back_yes','loan_paid_back_no'])

# use pydotplus graph from dot data to draw
graph_big = pydotplus.graph_from_dot_data(dot_data)


# show graph 
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


