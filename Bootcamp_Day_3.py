#!/usr/bin/env python
# coding: utf-8

# # Anshu Kumar

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\\Users\\anshu kumar\\Downloads\\Bootcamp_Assignment__4.ipynb")


# In[3]:


df.head()


# In[4]:


df


# In[6]:


df.shape


# # Data preprocessing

# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.corr()


# # EDA

# In[11]:


df.plot(kind='box')


# In[12]:


sns.scatterplot(x='Temperature (°C)', y='Ice Cream Sales (units)', data=df)
plt.show()


# In[13]:


temperature = df['Temperature (°C)']
sales = df['Ice Cream Sales (units)']
plt.bar(temperature, sales)
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.title('Ice Cream Sales vs Temperature')


plt.show()


# # TRAINING MODEL

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[16]:


X = df[['Temperature (°C)']]
y = df['Ice Cream Sales (units)']


# In[17]:


print("______________________________________________")
print(y.head())
print("______________________________________________")
print(X.head())


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


X_train.head()


# # Model building

# In[20]:


model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


print(y_pred)


# In[23]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# # Evaluation and Aaccuracy

# In[25]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[26]:


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.title('Linear Regression Model')
plt.show()

