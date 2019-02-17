
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

#The objective of this file is to read raw data into data frame.
# Read raw traing data all_X_train:
path='D:/ML/COMP551/comp-551-imbd-sentiment-classification/train/train/pos'
files=os.listdir(path)
all_X_train=pd.DataFrame({'A' : []})
for file in files:
    data = pd.read_table(path+"/"+file,header=None)
    all_X_train=pd.concat([all_X_train,data])

path='D:/ML/COMP551/comp-551-imbd-sentiment-classification/train/train/neg'
files=os.listdir(path)
for file in files:
    data = pd.read_table(path+"/"+file,header=None)
    all_X_train=pd.concat([all_X_train,data])
#save raw traing data
del all_X_train['A']
del all_X_train[1]
del all_X_train[2]
del all_X_train[3]
del all_X_train[4]
del all_X_train[5]
all_X_train.to_pickle('all_X_train.pkl')


# Read raw test data all_X_test:
import os
path='D:/ML/COMP551/comp-551-imbd-sentiment-classification/test/test'
files=os.listdir(path)
all_X_test=pd.DataFrame({'A' : []})
file_id=[] # The order of reading/File ID
for file in files:
    data = pd.read_table(path+"/"+file,header=None)
    all_X_test=pd.concat([all_X_test,data])
    file_id.append(file)
# Save raw test data:

