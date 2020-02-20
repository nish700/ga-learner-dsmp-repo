# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# load the dataset
df = pd.read_csv(path)

# print first 5 records
print(df.head())

# split the dataset into labels and features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.3)

#Now we will scale the records using MinMax scaler
scaler = MinMaxScaler()

# fit and transform the train data
X_train = scaler.fit_transform(X_train)
# transform the test data
X_test = scaler.transform(X_test)

# print the scaled values
print(X_train)
print(X_test)

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Instantiate the Logistic Regression model
lr = LogisticRegression()
# train the model
lr.fit(X_train, y_train)
# predict the labels
y_pred = lr.predict(X_test)

# calculate the roc_auc  score
roc_score = roc_auc_score = roc_auc_score(y_test, y_pred)

print('Area under the ROC AUC curve is:', roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier

# initialise the decision tree classifier
dt = DecisionTreeClassifier(random_state = 4)
# fit the model
dt.fit(X_train , y_train)

# predict
y_pred = dt.predict(X_test)

# calculate the roc auc score
roc_score = roc_auc_score(y_test, y_pred)
print('Area under the roc auc curve is :', roc_score)


# --------------
from sklearn.ensemble import RandomForestClassifier


# instantiate the random forest classifier
rfc = RandomForestClassifier(random_state = 4)

# fit the model
rfc.fit(X_train, y_train)

# predict
y_pred = rfc.predict(X_test)

# roc auc score
roc_score = roc_auc_score(y_test, y_pred)

# print
print('Area under the curve for ROC AUC curve using Random Forest classifier is:', roc_score)


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier

#initialise the bagging classifier
bagging_clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=100, max_samples=100, random_state=0)

# fit the model
bagging_clf.fit(X_train, y_train)

# calculate the score
score_bagging = bagging_clf.score(X_test, y_test)

print('Accuracy score of bagging classifier is :',score_bagging)


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


#initialise the voting classifier
voting_clf_hard = VotingClassifier(estimators = model_list, voting='hard')
# fit the model
voting_clf_hard.fit(X_train, y_train)
# calculate the score
hard_voting_score =  voting_clf_hard.score(X_test , y_test)

print('Accuracy of Voting Classifier using hard voting is:', hard_voting_score)


