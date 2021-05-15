#!/usr/bin/env python
# coding: utf-8

# # Author: Kodoori Akshay

# # Task-1 PREDICTION USING SUPERVISED LEARNING(ML)
# In this task, we'll predict the percentage  of marks that a student is expected to score based upon number of hours they've studied. This is a simple linear regression task,it is shown in two variables

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("C:/Users/Akshay/Desktop/spark.txt")


# In[3]:


data.head(5)


# In[4]:


data.info()


# In[5]:


data.dtypes


# In[6]:


data.describe()


# # Data Visualization & model fitting

# In[7]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
print(X)


# In[8]:


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[9]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.21, random_state=0) 


# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[11]:


print(X_train)


# In[12]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[13]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# # Actual vs Predicted

# In[14]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[15]:


X.ndim


# In[16]:


X.shape


# In[17]:


y.ndim


# In[18]:


y.shape


# In[19]:


hours =[[9.25],[5.0],[2.2]]
print(hours)


# # Testing our own Data

# In[20]:


# a= hours.reshape(1,-1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred))


# # Evaluating the model

# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Absolute Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




