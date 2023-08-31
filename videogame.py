#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


videog_data = pd.read_csv("Video_GSales_2016.csv")
videog_data.head()


# In[3]:


videog_data.shape


# In[4]:


videog_data.info()


# In[5]:


videog_data.tail(10)


# In[6]:


videog_data.isna().sum()


# In[7]:


videog_data= videog_data.dropna()


# In[8]:


videog_data.isna().sum()


# In[9]:


videog_data.head(10)


# In[10]:


training_data=videog_data.iloc[:,[2,5,6,7,8,10,11,12,13]]
testing_data =videog_data.iloc[:,[9]]


# In[11]:


videog_data.groupby("Year_of_Release")["Global_Sales"].agg(np.mean).plot(kind="bar")
plt.show()


# In[12]:


videog_data.groupby("Year_of_Release")["Global_Sales"].agg(np.mean).plot(kind="line")
plt.show()


# import matplotlib as mpl
# game = videog_data.groupby('Genre')['Global_Sales'].count().head(10)
# custom_colors = mpl.colors.Normalize(vmin = min(game), vmax=max(game))
# colors = [mpl.cm.PuBu(custom_colors(i)) for i in game]
# plt.figure(figsize=(7,7))
# plt.pie(game, labels=game.index, colors=colors)
# central_circle = plt.Circle((0,0), 0.5, color='white')
# fig = plt.get('obj')
# fig.gca().add_artist(central_circle)
# plt.rc('font', size=12)
# plt.title('Top 10 Categories of Games Sold', fontsize=20)
# plt.show()

# In[13]:


#EDA


# In[14]:


import matplotlib as mpl
game = videog_data.groupby("Genre")["Global_Sales"].count().head(10)
custom_colors = mpl.colors.Normalize(vmin=min(game), vmax=max(game))
colours = [mpl.cm.PuBu(custom_colors(i))for i in game]
plt.figure(figsize=(7,7))
plt.pie(game, labels=game.index, colors=colours)
central_circle = plt.Circle((0,0), 0.5, color="white")
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 10 Categories of Games Solid", fontsize=20)
plt.show()


# In[15]:


print(videog_data.corr())
sns.heatmap(videog_data.corr(), cmap="YlOrBr")
plt.show()


# In[16]:


x = videog_data[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales", "Critic_Score", "Critic_Count"]]
y = videog_data["User_Count"]


# In[17]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, random_state=42)


# In[18]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[19]:


predictions


# In[20]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[21]:


predictions


# 
