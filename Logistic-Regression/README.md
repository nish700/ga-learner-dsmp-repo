### Project Overview

 Logistic regression is another technique borrowed by machine learning from the field of statistics.

**Types of Logistic Regression**

1. Binary Logistic Regression

The categorical response has only two 2 possible outcomes. Example: Spam or Not

2. Multinomial Logistic Regression

Three or more categories without ordering. Example: Predicting which food is preferred more (Veg, Non-Veg, Vegan)

**Decision Boundary**

To predict which class a data belongs, a threshold can be set. Based upon this threshold, the obtained estimated probability is classified into classes.

**Decision boundary can be linear or non-linear. Polynomial order can be increased to get complex decision boundary.**

**ROC-AUC score**

Evaluate your classifier based on the AUC score according to:

•	.90-1 = excellent classifier

•	.80-.90 = good classifier

•	.70-.80 = fair classifier

•	.60-.70 = poor classifier

•	.50-.60 = fail classifier


**About the Dataset:**

The dataset has details of 1338 Insurance claim with the following 8 features.

**age**  age of policyholder

**sex**	male(1)/female(0)

**bmi	body** mass index(kg /m^2m2)

**children**	number of children/dependents of policyholder

**smoker**	smoking state nonsmoker(0)/smoker(1)

**region**	residential area northeast(0)/northwest(1)/southeast(2)/southwest(3)

**charges**	medical cost

**insuranceclaim**	yes(1)/no(0)


Following concepts have been implemented in the project:

•	**Train-test split**

•	**Correlation between the features**

•	**Logistic Regression**

•	**Auc score**

•	**Roc AUC plot**



