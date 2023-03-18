# In[1]:


# Importing the libraries

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Data collection and preprocessing

movies_data = pd.read_csv('movies.csv')


# In[3]:


movies_data.head()


# In[4]:


movies_data.shape


# In[5]:


# selecting the relevant features for recommendation

selected_features = ['genres', 'keywords', 'director', 'tagline', 'cast']
print(selected_features)


# In[6]:


# Replacing the null values with null string

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# In[7]:


# Combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


# Converting the textual data into feature vectors or numerical values

vectorizer = TfidfVectorizer()


# In[10]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[11]:


print(feature_vectors)


# In[12]:


# Using the Cosine Similarity to get the similiarity score

similarity = cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


print(similarity.shape)


# In[15]:


# Getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[16]:


# Creating a list with all the movies name given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[17]:


# Finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[18]:


close_match = find_close_match[0]
print(close_match)


# In[19]:


# Finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[20]:


# Getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[21]:


len(similarity_score)


# In[22]:


# Sorting the movies based on higher similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)


# In[23]:


# Print the name of similar movies based on th index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i<30):
        print(i, '.',title_from_index)
        i+=1


# In[24]:


# Saving the trained model

import pickle


# In[25]:


filename = 'trained_model.sav'
pickle.dump(similarity, open(filename, 'wb'))


# In[26]:


# Loading a saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[ ]:




