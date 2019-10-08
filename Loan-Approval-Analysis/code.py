# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

# code starts here

bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include='object')
print(categorical_var.head())
numerical_var = bank.select_dtypes(include='number')
print(numerical_var.head())
# code ends here


# --------------
# code starts here
banks = bank.drop('Loan_ID',axis=1)
# print(banks.columns)
# print(banks.head())

print(banks.isnull().sum())

bank_mode = mode(mode,axis=0)
print(bank_mode)

banks = banks.fillna(mode)

print(banks.isnull().sum())

#code ends here


# --------------
# Code starts here
print(banks.info())

avg_loan_amount = pd.pivot_table(banks, index = ['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc='mean')

print(avg_loan_amount)
# code ends here



# --------------
# code starts here

# print(banks.info())

# print(len(banks[banks['Loan_Status'].astype(str)=='Y' & banks['Self_Employed'].astype(str)=='Yes']))
mask_1 = (banks['Loan_Status'] =='Y')
mask_2 = (banks['Self_Employed'] == 'Yes')


loan_approved_se = len(banks[mask_1 & mask_2])

mask_3 = (banks['Loan_Status'] == 'Y')
mask_4 = (banks['Self_Employed']=='No')

loan_approved_nse = len(banks[mask_3 & mask_4])

# print(loan_approved_se)
# print(loan_approved_nse)

loan_status_count = 614

percentage_se = (loan_approved_se/loan_status_count) * 100
percentage_nse = (loan_approved_nse/loan_status_count) * 100

print(percentage_nse)
print(percentage_se)

# print(len(banks[banks['Loan_Status']=='Y']))
# print(len(banks[banks['Self_Employed']=='Yes' & banks['Loan_Status']=='Y']))

# code ends here


# --------------
# code starts here

# print(banks['Loan_Amount_Term'])
loan_term = banks['Loan_Amount_Term'].apply(lambda x:x/12)
print(len(loan_term))

big_loan_term = len(loan_term[loan_term >= 25])
print(big_loan_term)
# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')

loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]

mean_values = loan_groupby.apply(np.mean)

print(mean_values)
# code ends here


