#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[36]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Accuracy: {rf.score(X_test, y_test)}")


# In[40]:


get_ipython().system(' pip install xgboost')


# In[78]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}
gb_model = xgb.train(params, dtrain, num_boost_round=100)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"XGBoost Accuracy: {accuracy}")


# # Adaboost

# In[76]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train,y_train)
y_pred = ada.predict(X_test)
y_pred = ada.predict(X_test)
print(f"AdaBoost Accuracy: {ada.score(X_test, y_test)}")


# # Gradient Descent

# In[85]:


import numpy as np
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
m = b = 0
learning_rate = 0.1
epochs = 1000
for _ in range(epochs):
    y_pred = m*X + b 
    D_m = (-2/len(X)) * sum(X * (y - y_pred))
    D_b = (-2/len(X)) * sum(y - y_pred)
    m -= learning_rate * D_m
    b -= learning_rate * D_b
print(f"slope: {m}, intercept: {b}")   


# In[100]:


import pandas as pd

# Load the dataset
file_path = r"C:\\Users\\anshu kumar\\Downloads\\archive (2)\StressLevelDataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop duplicate rows
data = data.drop_duplicates()
print("\nData after removing duplicates:")
print(data.head())

# Fill missing values (example: fill with mean for numerical columns)
# Adjust this part based on your dataset's needs
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].mean(), inplace=True)

# Convert data types if necessary
# For example, if a column is incorrectly typed
# data['column_name'] = data['column_name'].astype('desired_type')

# Display cleaned data summary
print("\nCleaned Data Summary:")
print(data.describe())

# Save the cleaned data to a new CSV file
cleaned_file_path = r"C:\Users\anshu kumar\Downloads\archive (2)\Cleaned_StressLevelDataset.csv"
data.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned data saved to: {cleaned_file_path}")


# In[103]:


import pandas as pd


# In[104]:


from xgboost import XGBClassifier 


# In[105]:


from sklearn.model_selection import train_test_split


# In[106]:


from sklearn.metrics import accuracy_score


# In[107]:


from sklearn.impute import SimpleImputer 


# In[108]:


from sklearn.preprocessing import StandardScaler 


# In[109]:


file_path = r"C:\\Users\\anshu kumar\\Downloads\\archive (2)\\StressLevelDataset.csv"


# In[110]:


data = pd.read_csv(file_path)


# In[111]:


target_column = ['target_column']


# In[115]:


X = "data.drop(target_column, axis=1)"


# In[118]:


y = data['target_column']


# In[119]:


imputer = SimpleImputer(strategy='mean')


# In[120]:


X = imputer.fit_transform(X)


# In[121]:


scaler = StandardScaler()


# In[122]:


X = scaler.fit_transform(X)


# In[123]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[124]:


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')


# In[125]:


model.fit(X_train, y_train)


# In[126]:


y_pred = model.predict(X_test)


# In[127]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




