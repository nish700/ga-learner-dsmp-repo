# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data = pd.read_csv(path)

print(data.shape)

# print(data['Rating'].value_counts().sort_values())
plt.figure()

data['Rating'].hist(bins=5)

# sns.distplot(data['Rating'])

data = data[data['Rating']<= 5]

plt.figure()
data['Rating'].hist(bins=5)

# sns.distplot(data['Rating'])

#Code ends here


# --------------
# code starts here
## exploring the data for null values 

total_null = data.isnull().sum()

percent_null = total_null/ total_null.shape[0]

missing_data = pd.concat([total_null, percent_null], keys=['Total','Percent'], axis=1)

print(missing_data)

#removing the missing values
data.dropna(axis=0, inplace=True)

#verifying the presence of null values
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/total_null_1.shape[0]

missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])

print(missing_data_1)
# code ends here


# --------------

#Code starts here

g = sns.catplot(x='Category',y='Rating',kind='box',height=10, data=data)
g.set_xticklabels(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
plt.show()

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
# Clean the data, removing the + and , operator
print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].str.replace(',','')

#converting the data type to int
data['Installs'] = data['Installs'].astype(int)

# initializing the LabelEncoder
le = LabelEncoder()

# fit the train data to the Encoder
le.fit(data['Installs'])
# print the classes
print(list(le.classes_))
# transform the data and store in the dataset
data['Installs'] = le.transform(data['Installs'])

# print(data['Installs'].value_counts())

# ##plotting the tranformed data
sns.regplot(x='Installs',y='Rating',data=data)
plt.title('Rating vs Installs [RegPlot]')
# #Code ends here



# --------------
#Code starts here

# analysing the Price column 
print(data['Price'].dtype)
print(data['Price'].value_counts())

# cleaning $ sign from the price column and changing the data type to float
data['Price'] = data['Price'].str.replace("$",'')
data['Price'] = data["Price"].astype(float)

# plt the distribution to observe the correlation
sns.regplot(x='Price',y='Rating', data=data)
plt.title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
# analysing the genres column, the column has multiple genres for a phone
print("Unique values in Genres:",len(data['Genres'].unique()))
print(data['Genres'].head())

# cleaning the data and retaining only the first genre for an app
data['Genres'] = data['Genres'].str.split(pat=';', n=1, expand=True)[0]

#Group Genres and Rating by Genres
gr_mean = data[['Genres','Rating']].groupby(by=['Genres'], as_index= False).mean()

#Print the statistics of the group
print(gr_mean.describe())

# sorting the group mean by Rating
gr_mean = gr_mean.sort_values(by=['Rating'])

print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])

#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])

#converting the data type to datetime
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#Calculating the last updated in terms of number of days taking the max uapted date as #reference
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

# plotting the relation between Last Updated days and Rating
sns.regplot(x='Last Updated Days',y='Rating',data=data)
plt.title('Rating vs Last Updated [RegpPlot]')
# We see an inverse relationship between Last Updated Days and User Rating
#Code ends here


