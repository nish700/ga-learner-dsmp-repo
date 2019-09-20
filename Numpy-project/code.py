# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data = np.genfromtxt(path, delimiter=',' ,skip_header=1)

census = np.concatenate((data,np.asarray(new_record)))
print(census, type(census))


# --------------
#Code starts here
import numpy as np

age = census[:,0] # indexing an array, all rows and only the first column
max_age = max(age)
min_age = min(age)
age_mean = np.mean(age)
age_std = np.std(age)

print(age)
print(max_age)
print(min_age)
print(age_mean)
print(age_std)


# --------------
#Code starts here
race_0 = census[census[:,2] == 0]
race_1 = census[census[:,2] == 1]
race_2 = census[census[:,2] == 2]
race_3 = census[census[:,2] == 3]
race_4 = census[census[:,2] == 4]

len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)

len_race = [len_0,len_1,len_2,len_3,len_4]
minority_race = len_race.index(min(len_race))
print(minority_race)


# --------------
#Code starts here
senior_citizens = census[census[:,0]>60]

working_hours_sum = sum(senior_citizens[:,6])

senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum/senior_citizens_len

# print(working_hours_sum)
print(avg_working_hours)


# --------------
#Code starts here
import numpy as np

high = census[census[:,1]>10]
low = census[census[:,1]<=10]

avg_pay_high = np.mean(high[:,7])
avg_pay_low = np.mean(low[:,7])

print(avg_pay_high)
print(avg_pay_low)


