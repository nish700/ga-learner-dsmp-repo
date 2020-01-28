# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here

# load the dataset
dataset = pd.read_csv(path)


# analyse the first five rows
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset.drop(columns=['Id'], axis=1, inplace=True)

# check the statistical description
print(dataset.describe)



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)
size = dataset.shape[1] - 1


# split the dataset into features and labels
#x-axis has target attribute to distinguish between classes
#y-axis shows values of an attribute

x = dataset.drop(columns=['Cover_Type'], axis=1)
y = dataset['Cover_Type']

#Plot violin for all attributes
# for i in range(size):
#     sns.violinplot(x=cols[i],y='Cover_Type', data=dataset)

# alternate way of achieveing the violin plot
# fig, axes = plt.subplots(nrows=18, ncols=3,figsize=(12,8))
# # loop through the subplot to plot violin plot for all the features
# for i, row in enumerate(axes):
#     for j, ax in enumerate(row):
#         col = cols[i*3 + j]
#         sns.violinplot(x=col, y='Cover_Type', data=dataset, ax= axes[i,j])



# --------------
import numpy

upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
# subset the dataset to include only the first 10 features
subset_train = dataset.iloc[:,:10]

# calculate the Pearson correlation for the filtered features
data_corr = subset_train.corr()

# print(data_corr)

# plot the heatmap for the features
sns.heatmap(data_corr, cmap='YlGnBu')

# list the correlation pairs
correlation = data_corr.unstack().sort_values(kind='quicksort')

print(len(correlation))
# Slice and select values using the threshold mentioned above, value of 1 is not desirable so neglected
mask_1 = correlation > upper_threshold
mask_2 = correlation < lower_threshold
mask_3 = correlation != 1

corr_var_list = correlation[ (mask_1 | mask_2) & mask_3]

print(len(corr_var_list))
# Code ends here




# --------------
#Import libraries
import numpy as np 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

# split the dataset into train and test set
X = dataset.drop(columns=['Cover_Type'], axis=1)
Y = dataset['Cover_Type']

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, random_state=0, test_size=0.2)

# identifying the categorical and continuous features
continuous = X_train.iloc[:,:10].columns.tolist()
# select_dtypes(include=np.number).columns.tolist()
categorical = X_train.iloc[:,10:].columns.tolist()
# .select_dtypes(include=['object']).columns.tolist()

# print(categorical)
# print(continuous)
# print(X_train.info)
# print(X_train.dtypes)
# print(continuous)
# print(categorical)

#Apply transform only for continuous data
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train[continuous])
X_test_temp = scaler.transform(X_test[continuous])

print(len(X_train_temp))
print(len(X_test_temp))

# # #Concatenate scaled continuous data and categorical
X_train1 = np.concatenate((X_train_temp, X_train[categorical]), axis=1)
X_test1 = np.concatenate((X_test_temp, X_test[categorical]), axis=1)

print(X_train1)
print(X_test1)

scaled_features_train_df = pd.DataFrame(X_train1, columns = X_train.columns, index = X_train.index)
scaled_features_test_df = pd.DataFrame(X_test1 , columns = X_test.columns , index = X_test.index)

print(scaled_features_test_df.head())
print(scaled_features_train_df.head())


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

#initliaze SelectPercentile
skb = SelectPercentile(score_func = f_classif, percentile=90)

# fit and transform on train and test data
predictors = skb.fit_transform(X_train1, Y_train)

# calculate the score of different features
scores = skb.scores_

# # print(scores)
# print(predictors.shape)
# print(len(scores))

# save the feature names in a variable
Features = scaled_features_train_df.columns

dataframe = pd.DataFrame({'Features': Features, 'scores':scores})

# print(Features)
# print(dataframe)

dataframe.sort_values(ascending=False, by='scores', inplace=True)

print(dataframe.shape)

# print(dataframe['Features'][:predictors.shape[1]])
top_k_predictors = dataframe['Features'][:predictors.shape[1]].tolist()

print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

# initialize the One Vs Rest Classifier using Logistic regression as the classifer
clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

# fit the classifier for all features
model_fit_all_features = clf1.fit(X_train, Y_train)
# predict using all features
predictions_all_features = model_fit_all_features.predict(X_test)
# calculate the score
score_all_features = accuracy_score(Y_test,predictions_all_features)


# fit the classifier for only the top features
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
# predict using only the top features
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
#calculate the score
score_top_features = accuracy_score(Y_test, predictions_top_features)

# print the accuracy
print('Accuracy score for prediction using all features:',score_all_features)
print('Accuracy score for prediction using top features:',score_top_features)

# print(scaled_features_train_df[top_k_predictors])


