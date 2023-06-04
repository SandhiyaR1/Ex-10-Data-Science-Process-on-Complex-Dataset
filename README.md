# Ex-10-Data-Science-Process-on-Complex-Dataset
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
### STEP 1 :
Read the given Data

### STEP 2 :
Clean the Data Set using Data Cleaning Process

### STEP 3 :
Apply Feature Generation/Feature Selection Techniques on the data set

### STEP 4 :
Apply EDA /Data visualization techniques to all the features of the data set

### CODE:
Developed by: SANDHIYA R

Register No: 212222230129
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("/content/StudentsPerformance - StudentsPerformance.csv.csv")
print(data)
data.info()
data.isnull().sum()
# data cleaning

data['test preparation course']=data['test preparation course'].fillna(data['test preparation course'].mode()[0])

data['math score']=data['math score'].fillna(data['math score']).fillna(data['math score'].mean())

data['writing score']=data['writing score'].fillna(data['writing score']).fillna(data['reading score'].median())

data.isnull().sum()

data.describe()

data.head()
```

# removing outliers
```
Q1=data['math score'].quantile(0.25)

Q3=data['math score'].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR

upper=Q3+1.5*IQR

df=data[(data['math score']>=lower) & (data['math score']<=upper)] 
print(df) 


outliers=data[(data['math score']<lower) | (data['math score']>upper)] 
print(outliers)

df.shape
```
# Feature generation
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
df1=df.copy()
r=['group A','group B','group C','group D','group E']
enc=OrdinalEncoder(categories=[r])
enc.fit_transform(df1[['race/ethnicity']])
df1['neword1']=enc.fit_transform(df1[['race/ethnicity']])
df1 
df2=df1.copy()
le=LabelEncoder()
df2['neword2']=le.fit_transform(df2['race/ethnicity'])
df2
from sklearn.preprocessing import OneHotEncoder
df3=df.copy()
ohe=OneHotEncoder(sparse=False)
enc=pd.DataFrame(ohe.fit_transform(df3[['lunch']]))
df3=pd.concat([df3,enc],axis=1)
df3.head()
!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df4=df.copy()
newdata=be.fit_transform(df4['test preparation course'])
df4=pd.concat([df,newdata],axis=1)
df4.head()
```
# heatmap
```
data.corr()
plt.subplots(figsize=(7,5))
sns.heatmap(data.corr(),annot=True)
```
# Data visualization

# Scatter plot of math score vs. reading score
```

plt.scatter(data['math score'], data['reading score'])
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs. Reading Score')
plt.show()
sns.barplot(x='gender',y='reading score',data=df)
sns.boxplot(x="math score",data=df)
```
#  OUTPUT:
### DATA
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/d245ca8a-8feb-4af4-814f-52fa96c3a8dd)

### data.info()
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/d4efe78b-b9e9-4012-81d9-14f872721542)
### data.isnull.sum()
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/de162388-c6af-40fb-b721-7a6149921be7)
### After removing null values
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/d1c23afe-94b9-49a9-a586-3143a5a35f1b)
### data.discribe()
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/c12ae2e6-0f40-4c56-ac71-e75a73cc197a)
### data.head()
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/a415f62d-d940-4915-bc9d-cdf4af1224ab)
### New data after removing outliers
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/1807656e-00f7-4876-9d81-11214c0cf8e2)
### Outliers
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/3caa798c-759f-42c4-8a7e-24f724f5eed4)
### df.shape()
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/71d1a9f1-6d26-4b7d-9fb2-c47b1fa1b12e)
### Ordinal Encoding
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/54136182-8596-480a-9f00-f5379cf683e4)
### Label Encoding
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/c88d38d8-9614-45bd-9556-7759b6906f88)
### OneHot Encoding
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/69d91695-872b-459d-86e9-2011c0f5c33c)
### Binary Encoding
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/16f50bcd-0cef-483d-a6a3-ad75520b2ea0)
### Heatmap
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/e5094333-a584-45fa-9c52-e645a10398e2)
### Scatterplot
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/94fda449-5285-4091-b6d0-71f5af61fe1f)
### Barplot
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/3c7634a9-1556-41ab-a1e5-6095e9daf573)
### Boxplot
![image](https://github.com/SandhiyaR1/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/113497571/e7a4af4e-634c-4999-a47f-0d858a302e63)
# RESULT:
Hence, Data Science Process is performed on a complex dataset and saved the data to a file.
