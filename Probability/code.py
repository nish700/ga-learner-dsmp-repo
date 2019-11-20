# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)

# print(df.head())

#Calculate the probability p(A)for the event that fico credit
fico_score = df[df['fico']>700]
p_a = len(fico_score)/len(df['fico'])

#Calculate the probabilityp(B) for the event that purpose == 'debt_consolation'
p_b = len(df[df['purpose']=='debt_consolidation'])/len(df['purpose'])

#
df1 = df[df['purpose']=='debt_consolidation']

#probability of a and b
p_a_and_b = len(df[(df['purpose']=='debt_consolidation') & (df['fico']>700)])/len(df1)

print(p_a_and_b)

#Calculate the probablityp(B|A) for the event purpose == 'debt_consolidation' given 'fico' credit score is greater than 700
#probability of a given b

p_a_b = p_a_and_b/p_b

print(p_a_b)
print(p_a)

if (p_a_b == p_a):
    result= True
else:
    result= False

print(result)
# code ends here


# --------------
# code starts here

#Let's calculate the bayes theorem for the probability of credit policy is yes and the person is given the loan.

#Calculate the probability p(A) for the event that paid.back.loan == Yes
prob_lp = len(df[df['paid.back.loan']=='Yes'])/len(df['paid.back.loan'])

#Calculate the probability p(B) for the event that credit.policy == Yes
prob_cs = len(df[df['credit.policy']=='Yes'])/len(df['credit.policy'])

#
new_df = df[df['paid.back.loan']=='Yes']

#joint probability of both the conditions satisfied 
# prob_lp_and_cs = len(df[(df['paid.back.loan']=='Yes') & (df['credit.policy']=='Yes')])/len(new_df)

##Calculate the probablityp(B|A) for the event paid.back.loan == 'Yes' given credit.policy == 'Yes'
prob_pd_cs = new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]

##Calculate the conditional probability using the Bayes theorm
p_cs_pd = (prob_pd_cs * prob_lp)/prob_cs

bayes = p_cs_pd

print(bayes)
# ​
# code ends here


# --------------
# code starts here
#Let's visualize the bar plot for the purpose and again using condition where

#Visualize the bar plot for the feature purpose
fig = plt.figure(figsize=(15,50))
ax_1 = plt.subplot(3,1,1)
ax_2 = plt.subplot(3,1,2)
ax_3 = plt.subplot(3,1,3)

df['purpose'].value_counts().plot(kind='bar',ax=ax_1)

#Visualize the bar plot for the feature purpose where paid.back.loan == No
df1 = df[df['paid.back.loan']=='No']
df1['purpose'].value_counts().plot(kind='bar',ax= ax_2)

# lets check for the percentage of defaulters in each category
df2 = df1['purpose'].value_counts()/df['purpose'].value_counts()
df2.round(2).value_counts().plot(kind='bar',ax=ax_3)
# code ends here


# --------------
# code starts here
import seaborn as sns
#Let's plot the histogram for visualization of the continuous variable.

fig,(ax_1,ax_2) = plt.subplots(2,1,figsize=(20,20))

inst_median = df['installment'].median()

inst_mean = df['installment'].mean()

#plotting a histogram for the distribution
ax_1.hist(df['installment'],color='c',edgecolor='k',alpha=0.65,normed=True)

#marking mean and median on the histogram plot
ax_1.axvline(inst_mean,color='k',linestyle='dashed',linewidth=2)
ax_1.axvline(inst_median,color='r',linestyle='dashed',linewidth=2)

##drawing density plot over the histogram
sns.kdeplot(df['installment'],color='black',label='Frequency',ax=ax_1)

##making the tick size bigger
plt.setp(ax_1.get_xticklabels(),fontsize=14)
plt.setp(ax_1.get_yticklabels(),fontsize=14)

#plotting the histogram for log.annual.inc
ax_2.hist(df['log.annual.inc'],color='b',edgecolor='k',alpha=0.65,normed=True)
plt.setp(ax_2.get_xticklabels(),fontsize=14)
plt.setp(ax_2.get_yticklabels(),fontsize=14)

#display the plot
plt.show()
# code ends here


