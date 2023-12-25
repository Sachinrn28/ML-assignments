#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# Estimate the total compansation to be provided to an employe

# In[1]:


# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[210]:


# Importing data
data=pd.read_csv(r'E:\Data Is Everything\Edyoda\train_set.csv')
data.head(50)


# In[211]:


# The column names are not understandable we need to change.
col_name={'OGC':'Organization Group Code','OG':'Organization group name','DC':'Department code','Dept':'Department name','UC':'Union code','Union':'Union name','JF':'Job Family','Job':'Job name','EI':'Employee Identifier','H/D':'Health Dental','YT':'Year Type'}
data.rename(columns=col_name,inplace=True)


# In[70]:


data.sample(4)


# In[212]:


# Checking for null values
data.isnull().sum()


# In[213]:


# We can drop some of the columns which are not required for my analysis
name=['Year','Organization Group Code','Organization group name','Department code', 'Department name','Union code', 'Union name','Job Family','Job name','Year Type','Employee Identifier']
df=data.drop(columns=name)
df


# In[59]:


data.columns


# In[60]:


# Checking the shape of the data
df.shape


# # Observation
# - There are 287836 employes
# - 4 Columns

# In[61]:


# To know the data type 
df.info()


# # All data we have are integer and float

# In[73]:


# Checking for colinearity
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')


# # Observation
# * There is more correlation betwen the salaries and health Dental so i can drop that

# In[214]:


df=df.drop(columns='Health Dental')
df


# In[161]:


# Checking for outliers now
fig,axis=plt.subplots(3,figsize=(6,6))
sns.boxplot(x=df['Salaries'],ax=axis[0])
sns.boxplot(x=df['Overtime'],ax=axis[1])
sns.boxplot(x=df['Total_Compensation'],ax=axis[2])
plt.tight_layout()
plt.show()


# # Observation 
# - My data has negetive values they all should go positive
# - There is more than 200000 rupees that paid for overtime we can drop the column

# In[215]:


neget_val=df[(df['Salaries']<0) &(df['Total_Compensation']<0)].index
df=df.drop(index=neget_val)


# In[216]:


#plt.subplots(2,figsize=(6,6))
#sns.boxplot(x=df['Salaries'],ax=axis[0])
#sns.boxplot(x=df['Total_Compensation'],ax=axis[1])
#sns.boxplot(x=df['Salaries'],ax=axis[0])


# In[163]:


fig,axis=plt.subplots(2,figsize=(6,6))
sns.boxplot(x=df['Salaries'],ax=axis[0])
sns.boxplot(x=df['Total_Compensation'],ax=axis[1])


# In[217]:


# Dealing the outlier in the Salary column
sal=df['Salaries']
df['Salaries']=np.where(sal<(Q1-1.5*IQR),IQR,df["Salaries"])
df['Salaries']=np.where(sal>(Q3+1.5*IQR),IQR,df["Salaries"])
sns.boxplot(df['Salaries'])


# In[218]:


Q1=df['Salaries'].quantile(0.25)
Q3=df['Salaries'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
print(Q1)
print(Q3)


# In[219]:


# Dealing the outliers in column Total_Compensation
QQ1=df['Total_Compensation'].quantile(0.25)
QQ3=df['Total_Compensation'].quantile(0.75)
IQR2=QQ3-QQ1


# In[220]:


tot=df['Total_Compensation']
df['Total_Compensation']=np.where((tot<QQ1-1.5*IQR2),IQR,df['Total_Compensation'])
df['Total_Compensation']=np.where((tot>QQ3+1.5*IQR2),IQR,df['Total_Compensation'])


# In[192]:


sns.boxplot(x=df['Total_Compensation'])


# In[221]:


df=df.drop(columns='Overtime')


# In[222]:


df.sample(10)


# In[224]:


# Feater scaling
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
mms


# In[230]:


df['Total_Compensation']=mms.fit_transform(df[['Total_Compensation']])
df['Salaries']=mms.fit_transform(df[['Salaries']])


# In[232]:


df.sample(6)


# In[273]:


# Seperate x and y 
x=df.drop('Total_Compensation',axis=1)
y=df['Total_Compensation']


# In[274]:


# Spliting data fro train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=45,test_size=0.3)


# In[275]:


# Applying linear regerssion
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr


# In[276]:


lr.fit(x_train,y_train)


# In[277]:


y_pred=lr.predict(x_test)
y_pred


# In[247]:


x_test


# In[278]:


#Check the accuracy of the model now
from sklearn.metrics import r2_score
r2_score(x_test,y_test)


# In[279]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(x_test,y_test))


# In[280]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(x_train,y_train))


# # Observation
# - Our model is not biased

# In[290]:


# coordinates
lr.coef_


# In[291]:


lr.intercept_

