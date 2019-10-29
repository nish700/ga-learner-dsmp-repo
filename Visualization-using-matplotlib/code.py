# --------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()

axes = plt.figure().add_subplot(111)

loan_status.plot(kind='bar')
a = axes.get_xticks().tolist()

a[0]='Approved'
a[1]='Rejected'

axes.set_xticklabels(a)

plt.ylabel('No of loans')
plt.xlabel('Approval Status')
plt.title('Loan Approval Status')
plt.xticks(rotation=0)
plt.show()


# --------------
#Code starts here
property_and_loan = data.groupby(['Property_Area','Loan_Status'])

property_and_loan = property_and_loan.size().unstack()

property_and_loan.plot(kind='bar',stacked=False,figsize=(12,10))

plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here
education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()

education_and_loan.plot(kind='bar',stacked=True,figsize=(12,10))

plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here
graduate = data[data['Education']=='Graduate']
not_graduate = data[data['Education']=='Not Graduate']

graduate['LoanAmount'].plot(kind='density',label='Graduate')
not_graduate['LoanAmount'].plot(kind='density',label='Not Graduate')


#Code ends here

#For automatic legend display
plt.legend()
plt.show()


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(nrows=3,ncols=1,figsize=(20,15))

data.plot.scatter('ApplicantIncome','LoanAmount',ax=ax_1)
plt.title('Applicant Income')

data.plot.scatter('CoapplicantIncome','LoanAmount',ax=ax_2)
plt.title('Coapplicant Income')

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

data.plot.scatter('TotalIncome','LoanAmount',ax=ax_3)
plt.title('Total Income')

plt.show()


