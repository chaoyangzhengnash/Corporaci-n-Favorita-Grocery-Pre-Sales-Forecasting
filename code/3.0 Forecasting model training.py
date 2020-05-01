#
# Program: 3.0 Forecasting model training
#
# Purpose: Train Nearest neighbor, Liner regression and Neural network
#
# Written by: Qiuhua Liu(11258799) 
#             Yanhan Peng(11125583) 
#             Chaoyang Zheng(11249259) 
# 
# Updated: Dec 2019
#        
# ------------------------------------------------------.


############################
## Train Nearest neighbor ##
############################


# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import datetime


# In[2]:


data_train = pd.read_csv(r"/Users/yanhanpeng/Desktop/favorita-grocery-sales-forecasting/items_for_clustering.csv")


# In[3]:


new_data_train=pd.DataFrame({'item_nbr':data_train['item_nbr'],'store_nbr':data_train['store_nbr'],'store_city':data_train['store_city'],'store_state':data_train['store_state'],'store_type':data_train['store_type'],'store_cluster':data_train['store_cluster'],'item_family':data_train['item_family'],'item_class':data_train['item_class'],'item_perishable':data_train['item_perishable'],})


# In[4]:


new_data_train.head()


# In[5]:


new_data_train.info()


# In[17]:


new_data_train[new_data_train['item_nbr'].isin([1083152])]


# In[18]:


a=new_data_train.ix[51084]


# In[19]:


print(a)


# In[24]:


new_data_train.drop(index=51084,inplace=True)


# In[26]:


new_data_train.head()


# In[27]:


new_data_train = new_data_train.reset_index(drop=True)


# In[28]:


index=list(range(0,117618))


# In[29]:


min_data = {'distance':1,'index':None}


# In[32]:


print(min_data)


# In[33]:


from scipy.spatial import distance


# In[34]:


for n in index:
    distance_hamming=distance.hamming(a,new_data_train.ix[n])
    print(distance_hamming,n)
    if distance_hamming < min_data['distance']:
        min_data['distance'] = distance_hamming
        min_data['index'] =n
print(min_data)


# In[35]:


b=new_data_train.ix[3381]


# In[36]:


print(b)

