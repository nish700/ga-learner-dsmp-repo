# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)

data['Gender'].replace('-','Agender',inplace=True)

gender_count = data['Gender'].value_counts()

plt.figure(figsize=(10,6))
gender_count.plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Count of Gender')
plt.title('Gender Count')
plt.xticks(rotation=0)

#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()

explode = [0,0.1,0]
labels = alignment.index

fig,ax = plt.subplots()
ax.pie(alignment,explode = explode,labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Character Alignment')


# --------------
#Code starts here
## Calculating pearson correlation coeffcient for Strenght and Combat
sc_df = data[['Strength','Combat']]

sc_covariance = sc_df.cov().iloc[0,1]

sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()

## p = cov(x,y)/std(x)*std(y)

sc_pearson = sc_covariance/(sc_strength * sc_combat)

print(sc_pearson)

##Calculating Pearson correlation coefficient for Intelligence and Combat
ic_df = data[['Intelligence','Combat']].copy()

ic_covariance = ic_df.cov().iloc[0,1]

ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()

## p = cov(x,y)/std(x)*std(y)
ic_pearson = ic_covariance/(ic_intelligence * ic_combat)

print(ic_pearson)



# --------------
#Code starts here

## Who are the best of the best in this superhero universe? Let's find out.

total_high = data['Total'].quantile(q=0.99)

super_best = data[data['Total'] > total_high]

super_best_names = super_best['Name'].tolist()

print(super_best_names)


# --------------
#Code starts here

# Of the top 1% members of 'ASB', we want to measure certain attributes in case they go rogue and become threatening to the human kind.

fig, (ax_1, ax_2, ax_3) = plt.subplots(3,1,figsize=(15,30))

data.boxplot(column='Intelligence',ax=ax_1)
ax_1.set_title('Intelligence')

data.boxplot(column='Speed',ax=ax_2)
ax_2.set_title('Speed')

data.boxplot(column='Power',ax=ax_3)
ax_3.set_title('Power')

plt.show()




