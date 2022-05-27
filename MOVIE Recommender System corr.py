#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[ ]:


##Getting the Data


# In[4]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("C:/Users/Swetha/Documents/Movie-Recommender-in-python-corr/u.data", sep='\t', names=column_names)


# In[5]:


df.head()


#  ##Movie titles

# In[7]:


movie_titles = pd.read_csv("C:/Users/Swetha/Documents/Movie-Recommender-in-python-corr/Movie_Id_Titles")
movie_titles.head()


# merging them together

# In[8]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # EDA
# 
# Let's explore the data a bit and get a look at some of the best rated movies.
# 
# ## Visualization Imports

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's create a ratings dataframe with average rating and number of ratings:

# In[10]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[11]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[12]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Setting the number of ratings column

# In[13]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Now a few histograms:

# In[14]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[15]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[16]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# ## Recommending Similar Movies

# Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[17]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[18]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

# In[19]:


ratings.head()


# user ratings for those two movies is taken here 

# In[20]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


#  corrwith() method is used to get correlations between two pandas series

# In[21]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# Cleaning this by removing NaN values and using a DataFrame instead of a series

# In[22]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# In[23]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[24]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Sorting the values and notice how the titles make a lot more sense

# In[25]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# For the comedy Liar Liar

# In[26]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:




