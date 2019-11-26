# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here

##load the data
data = pd.read_csv(path)
# print(data.head())

##create a sample of the data
data_sample = data.sample(n=sample_size,random_state=0)

# print(data_sample.head())

##calculate the mean and save it in a variable
sample_mean = data_sample['installment'].mean()

##calculate the std and save it
sample_std = data_sample['installment'].std()

##calculate the margin of error
margin_of_error = (z_critical * sample_std)/math.sqrt(sample_size)

print("margin of error is:{0} ".format(margin_of_error))

##calculate the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

## calculate the true mean
true_mean = data['installment'].mean()

##inference
# print(true_mean)

# print(confidence_interval[0])
# print(confidence_interval[1])

if(true_mean> confidence_interval[0] and true_mean<confidence_interval[1]):
    print("true mean {0} falls within the confidence interval{1}".format(true_mean,confidence_interval))
else:
    print("true mean {0} does not fall within confidence interval {1}".format(true_mean,confidence_interval))



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here

##plotting histogram depicting the observance of Central Limit Theorm

fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(12,10))

for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        sample_data = data['installment'].sample(n=sample_size[i])
        sample_mean = sample_data.mean()
        m.append(sample_mean)
    mean_series = pd.Series(m)
    ##plotting the histogram of the mean for various sample size
    axes[i].hist(mean_series,color='b',edgecolor='k',alpha=0.65,normed=True)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here

data['int.rate'].replace(to_replace='%',value='',inplace=True,regex=True)

data['int.rate'] = data['int.rate'].astype(float)

data['int.rate'] = data['int.rate']/100

##calculate the z-score and p-value using z-test
z_statistic,p_value= ztest(data[data['purpose']=='small_business']['int.rate'],value= data['int.rate'].mean(),alternative='larger')

##making inference out of the p-value
if (p_value<0.05):
    inference = "Reject the Null Hypothesis"
else:
    inference = "Can't Reject the Null Hypothesis"

print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value = ztest(x1= data[data['paid.back.loan']=='No']['installment'],x2 = data[data['paid.back.loan']=='Yes']['installment'])

##print the statistic value
print(z_statistic)
print(p_value)

##deriving inference from p-value, if p-value less than 0.05O(lies in the Rejection Zone) --> Reject the Null hypothesis
if p_value<0.05:
    inference = 'Reject the NUll Hypothesis'
else:
    inference = 'Can''t reject the null hypothesis'

##print the inference
print(inference)



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()

# make a frequency table
observed = pd.concat([yes.transpose(),no.transpose()],axis=1,keys=['Yes','No'])

## conduct the chi-square test with the above frequency table
chi2, p ,dof,ex = stats.chi2_contingency(observed)

#draw inference from chi2 and critical_value
if(chi2 > critical_value):
    inference='Reject the Null Hypothesis'
else:
    inference= 'Can''t Reject the Null Hypothesis'

#print the inference
print(inference)




