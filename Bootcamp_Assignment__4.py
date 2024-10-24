#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


df = pd.read_csv("C:\\Users\\anshu kumar\\Downloads\\Bootcamp_Assignment__4.ipynb")


# In[ ]:


df.head(5)


# In[ ]:


df.tail(5)


# In[ ]:


df.columns


# In[ ]:


df=df.rename(columns={"sex":"gender"})


# In[ ]:


df.shape


# In[ ]:


df.size


# In[ ]:


df.isnull().sum()


# In[ ]:


df["region"].value_counts()


# In[ ]:


df["children"].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df["sex"]=labelencoder.fit_transform(df["gender"])
df["smoker"]=labelencoder.fit_transform(df["smoker"])
df["region"]=labelencoder.fit_transform(df["region"])


# In[ ]:


df


# In[ ]:


sns.histplot(df["age"])


# In[ ]:


sns.histplot(df["bmi"],kde = True)


# In[ ]:


sns.barplot(x="age",data=df,hue="smoker")


# In[ ]:


sns.barplot(x="gender",data=df,hue="smoker")


# In[ ]:





# In[ ]:


x= df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


print(x)


# In[ ]:


print(y)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[ ]:


print(x_train)


# In[ ]:


print(y_train)


# In[ ]:


svm_model = SVC(kernel='linear')


# In[ ]:


svm_model.fit(x_train, y_train)


y_pred = svm_model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)


# In[ ]:




