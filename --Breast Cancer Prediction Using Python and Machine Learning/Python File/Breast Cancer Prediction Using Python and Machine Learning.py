"""

@author Abubakar Sattar

"""

#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Prediction Using Python

# In[1]:


# importing libraries
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


# reading data from the file
df = pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#return all the columns with null values count
df.isna().sum()


# In[6]:


# returns the size of dataset
df.shape


# In[7]:


# removing the last column
df = df.dropna(axis = 1)


# In[8]:


# shape of dataset after removing the null column
df.shape


# In[9]:


# descrive the dataset
df.describe()


# In[10]:


# get the count of Malignant<M> and Benign<B>
df['diagnosis'].value_counts()


# In[11]:


sns.countplot(df['diagnosis'], label = "count")


# In[12]:


# label encoding(convert the values of M and B into 1 and 0)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)


# In[13]:


df.head()


# In[16]:


# effect of first three columns on diagnosis
sns.pairplot(df.iloc[:,1:5], hue = "diagnosis")


# In[17]:


# get the correlation
df.iloc[:,1:32].corr()


# In[19]:


# visualize the correlation
plt.figure(figsize = (10,10))
sns.heatmap(df.iloc[:,1:10].corr(), annot = True, fmt = ".0%") 


# In[20]:


# split the dataset into dependent(X) and independent(Y) datasets
X=df.iloc[:,2:32].values
Y=df.iloc[:,1].values


# In[22]:


print(X)


# In[24]:


# spliting the data into trainning and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[25]:


# feature scaling
from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# In[28]:


# Models/ Algorithms
def models(X_train,Y_train):
    # logistic regression Model
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train,Y_train)
    
    
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state = 0, criterion = "entropy")
    tree.fit(X_train,Y_train)
    
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(random_state = 0, criterion = "entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    
    print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
    print('[1]Decision Tree accuracy:',tree.score(X_train,Y_train))
    print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
    
    return log,tree,forest


# In[29]:


model = models(X_train,Y_train)


# In[30]:


# testing the models/results

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy : ',accuracy_score(Y_test,model[i].predict(X_test)))


# In[34]:


# prediction of random-forest
pred= model[2].predict(X_test)
print('Predicted Values')
print(pred)
print('Actual Values')
print(Y_test)


# In[36]:


from joblib import dump
dump(model[2],"Cancer_prediction_model;.joblib")


# In[ ]:




